plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "com.samanramezani.aichattest"
    compileSdk = 36

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
            // Do not use pickFirst for libonnxruntime.so: the Microsoft AAR is
            // unpacked as Java classes only, so sherpa's ABI-matched native
            // runtime is the only ONNX Runtime .so packaged in the APK.
            // Keep native libraries uncompressed and avoid a second extracted copy on disk.
            useLegacyPackaging = false
        }
    }
}

// Resolve ONNX Runtime's Java API without packaging its competing native .so.
 // sherpa-onnx bundles the native ONNX Runtime version required by libsherpa-onnx-jni.so.
val onnxRuntimeApiAar by configurations.creating {
    isCanBeConsumed = false
    isCanBeResolved = true
}

val onnxRuntimeApiJar = layout.buildDirectory.file("generated/onnxruntime-api/classes.jar")
val extractOnnxRuntimeApi by tasks.registering {
    inputs.files(onnxRuntimeApiAar)
    outputs.file(onnxRuntimeApiJar)
    doLast {
        val aar = onnxRuntimeApiAar.singleFile
        val output = onnxRuntimeApiJar.get().asFile
        output.parentFile.mkdirs()
        java.util.zip.ZipFile(aar).use { zip ->
            val entry = zip.getEntry("classes.jar")
                ?: error("ONNX Runtime AAR does not contain classes.jar: ${aar.name}")
            zip.getInputStream(entry).use { input ->
                output.outputStream().use { destination -> input.copyTo(destination) }
            }
        }
    }
}

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
    // The AAR supplies sherpa's Kotlin/JVM API and its matching native runtime.
    implementation("com.github.k2-fsa:sherpa-onnx:v1.13.8") {
        exclude(group = "com.github.k2-fsa.sherpa-onnx", module = "sherpa-onnx-jvm")
    }
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
