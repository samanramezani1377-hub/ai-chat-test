import java.net.URL
import java.security.MessageDigest
import java.util.zip.ZipFile

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
    id("org.mozilla.rust-android-gradle.rust-android") version "0.10.0"
}

android {
    namespace = "com.samanramezani.aichattest"
    compileSdk = 36
    ndkVersion = "27.2.12479018"

    defaultConfig {
        applicationId = "com.samanramezani.aichattest"
        minSdk = 29
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
        // The native inference runtime is built only for arm64-v8a. Restrict the
        // app package too, otherwise sherpa-onnx contributes four ABI copies
        // (arm64, 32-bit ARM, x86, x86_64) and bloats the APK without adding
        // runnable support for those extra ABIs.
        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildTypes {
        getByName("release") {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    packagingOptions {
        resources {
            // sherpa-onnx's AAR also ships macOS and Windows runtime binaries as
            // Java resources. They are never loaded by Android and add tens of
            // megabytes to the APK; keep the Android JNI libraries untouched.
            excludes += setOf("sherpa-onnx/native/**")
        }
        jniLibs {
            // ONNX Runtime and its JNI bridge come from the same Microsoft AAR;
            // sherpa uses its static-link AAR to avoid a duplicate SONAME.
            // Keep native libraries uncompressed and avoid a second extracted copy on disk.
            useLegacyPackaging = false
        }
    }
}


/*
 * Build Hugging Face's official Rust Tokenizers on-device for Laya.
 * Only arm64-v8a is packaged because this app's native inference runtime is arm64-only.
 */
cargo {
    module = "../rs-laya-tokenizers"
    libname = "laya_tokenizers"
    targets = listOf("arm64")
    profile = "release"
}

// Ensure the JNI tokenizer is compiled before Android packages native libraries.
tasks.named("preBuild").configure {
    dependsOn("cargoBuild")
}

// sherpa-onnx's regular AAR exports ONNX Runtime as libonnxruntime.so, which
// collides with the separate Microsoft JNI bridge used by Laya. Use sherpa's
// official static-link AAR so ASR/TTS carries its own statically linked ORT,
// and reserve libonnxruntime.so for the matching Microsoft Java API/JNI pair.
val sherpaStaticAar = layout.buildDirectory.file("generated/sherpa/sherpa-onnx-static-link-onnxruntime-1.13.8.aar")
val downloadSherpaStaticAar by tasks.registering {
    outputs.file(sherpaStaticAar)
    doLast {
        val output = outputs.files.singleFile
        output.parentFile.mkdirs()
        URL("https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.8/sherpa-onnx-static-link-onnxruntime-1.13.8.aar")
            .openStream().use { input ->
                output.outputStream().use { destination -> input.copyTo(destination) }
            }
        val actual = MessageDigest.getInstance("SHA-256").digest(output.readBytes())
            .joinToString("") { "%02x".format(it) }
        check(actual == "b22c3fc1b6a45666d28892bb2f7694beeb77a8362d7ebd77c1a5431ec9435471") {
            "sherpa static-link AAR checksum mismatch: $actual"
        }
    }
}

// Resolve ONNX Runtime's Java API and both matching native libraries from one AAR.
val onnxRuntimeApiAar by configurations.creating {
    isCanBeConsumed = false
    isCanBeResolved = true
}

val onnxRuntimeApiJar = layout.buildDirectory.file("generated/onnxruntime-api/classes.jar")
val onnxRuntimeJniDir = layout.buildDirectory.dir("generated/onnxruntime-jni")
val extractOnnxRuntimeApi by tasks.registering {
    // Declare all task I/O through Gradle's task model. Do not capture the
    // script-level Configuration or Provider inside doLast: that breaks the
    // configuration cache used by CI.
    inputs.files(onnxRuntimeApiAar)
    outputs.file(onnxRuntimeApiJar)
    doLast {
        val aar = inputs.files.singleFile
        val output = outputs.files.singleFile
        output.parentFile.mkdirs()
        ZipFile(aar).use { zip ->
            val entry = zip.getEntry("classes.jar")
                ?: error("ONNX Runtime AAR does not contain classes.jar: ${aar.name}")
            zip.getInputStream(entry).use { input ->
                output.outputStream().use { destination -> input.copyTo(destination) }
            }
        }
    }
}

// Package the native runtime and JNI bridge from the same Microsoft 1.28.0
// AAR. Do not mix its JNI bridge with sherpa's separately-built ORT runtime.
val extractOnnxRuntimeJni by tasks.registering {
    inputs.files(onnxRuntimeApiAar)
    outputs.dir(onnxRuntimeJniDir)
    doLast {
        val aar = inputs.files.singleFile
        val outputDir = outputs.files.singleFile
        val abiDir = outputDir.resolve("arm64-v8a")
        abiDir.mkdirs()
        ZipFile(aar).use { zip ->
            listOf("libonnxruntime.so", "libonnxruntime4j_jni.so").forEach { library ->
                val path = "jni/arm64-v8a/$library"
                val entry = zip.getEntry(path) ?: error("ONNX Runtime AAR does not contain $path: ${aar.name}")
                zip.getInputStream(entry).use { input ->
                    abiDir.resolve(library).outputStream().use { destination -> input.copyTo(destination) }
                }
            }
        }
    }
}

android.sourceSets.getByName("main").jniLibs.srcDir(onnxRuntimeJniDir)
tasks.named("preBuild").configure { dependsOn(extractOnnxRuntimeJni, downloadSherpaStaticAar) }

dependencies {
    add(onnxRuntimeApiAar.name, "com.microsoft.onnxruntime:onnxruntime-android:1.28.0@aar")
    implementation(files(onnxRuntimeApiJar).builtBy(extractOnnxRuntimeApi))
    implementation(project(":core:domain"))
    implementation(project(":core:actions"))
    implementation(project(":core:observability"))
    implementation(project(":core:settings"))
    implementation(project(":core:runtime"))
    implementation(project(":core:runtime-android"))
    implementation(project(":core:conversation"))
    implementation(project(":core:agent"))

    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")
    // Local-only ASR/TTS engines; no network inference or cloud API.
    // Static-link variant avoids the libonnxruntime.so SONAME collision with
    // the independently initialized ONNX Runtime Java API used by Laya.
    implementation(files(sherpaStaticAar).builtBy(downloadSherpaStaticAar))
    implementation("org.apache.commons:commons-compress:1.28.0")

    // Keep all Compose artifacts on Google's stable BOM so Gradle cannot resolve
    // unversioned Compose modules to an incompatible alpha release.
    val composeBom = platform("androidx.compose:compose-bom:2026.06.01")
    implementation(composeBom)
    androidTestImplementation(composeBom)

    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.foundation:foundation-layout")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.compose.ui:ui-tooling-preview")
    debugImplementation("androidx.compose.ui:ui-tooling")
    debugImplementation("androidx.compose.ui:ui-test-manifest")

    implementation("androidx.activity:activity-compose:1.10.1")
    implementation("androidx.core:core-ktx:1.17.0")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")
    testImplementation("junit:junit:4.13.2")
}
