import java.net.URL
import java.security.MessageDigest
import java.util.zip.ZipFile

plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

// Qwen3.5 support was added to the model builder in newer ORT GenAI releases.
// Keep this aligned with the native Android AAR and adapter diagnostics.
val genAiVersion = "0.15.2"
val genAiAar = layout.buildDirectory.file("generated/onnxruntime-genai/onnxruntime-genai-android-$genAiVersion.aar")
val genAiJar = layout.buildDirectory.file("generated/onnxruntime-genai/classes.jar")
val genAiJniDir = layout.buildDirectory.dir("generated/onnxruntime-genai/jniLibs")

val downloadGenAiAar by tasks.registering {
    outputs.file(genAiAar)
    doLast {
        val output = outputs.files.singleFile
        output.parentFile.mkdirs()
        URL("https://github.com/microsoft/onnxruntime-genai/releases/download/v$genAiVersion/onnxruntime-genai-android-$genAiVersion.aar")
            .openStream().use { input -> output.outputStream().use { destination -> input.copyTo(destination) } }
        val actual = MessageDigest.getInstance("SHA-256").digest(output.readBytes())
            .joinToString("") { "%02x".format(it) }
        check(actual == "f507ddbf0f635554ab12fcbf7dcfb115981eb664b925d90f28ed598dd8eb2ee7") {
            "ONNX Runtime GenAI AAR checksum mismatch: $actual"
        }
    }
}

val extractGenAiAar by tasks.registering {
    dependsOn(downloadGenAiAar)
    inputs.file(genAiAar)
    outputs.file(genAiJar)
    outputs.dir(genAiJniDir)
    doLast {
        val aar = inputs.files.singleFile
        val jar = genAiJar.get().asFile
        val jniRoot = genAiJniDir.get().asFile
        jar.parentFile.mkdirs()
        jniRoot.mkdirs()
        ZipFile(aar).use { zip ->
            val classes = zip.getEntry("classes.jar")
                ?: error("ONNX Runtime GenAI AAR does not contain classes.jar")
            zip.getInputStream(classes).use { input -> jar.outputStream().use { output -> input.copyTo(output) } }
            val nativeEntries = zip.entries().asSequence().filter {
                !it.isDirectory && it.name.startsWith("jni/arm64-v8a/") && it.name.endsWith(".so")
            }.toList()
            check(nativeEntries.any { it.name.endsWith("libonnxruntime-genai.so") }) {
                "ONNX Runtime GenAI AAR is missing libonnxruntime-genai.so for arm64-v8a"
            }
            check(nativeEntries.any { it.name.endsWith("libonnxruntime-genai-jni.so") }) {
                "ONNX Runtime GenAI AAR is missing libonnxruntime-genai-jni.so for arm64-v8a"
            }
            nativeEntries.forEach { entry ->
                val output = jniRoot.resolve("arm64-v8a").resolve(entry.name.substringAfterLast('/'))
                output.parentFile.mkdirs()
                zip.getInputStream(entry).use { input -> input.outputStreamTo(output) }
            }
        }
    }
}

private fun java.io.InputStream.outputStreamTo(output: java.io.File) {
    output.outputStream().use { destination -> copyTo(destination) }
}

android {
    namespace = "com.woogit.aicore.runtime.android"
    compileSdk = 36

    defaultConfig {
        minSdk = 29
        consumerProguardFiles("consumer-rules.pro")
        ndk { abiFilters += "arm64-v8a" }
    }

    sourceSets.getByName("main").jniLibs.srcDir(genAiJniDir)

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions { jvmTarget = "17" }
}

tasks.named("preBuild").configure { dependsOn(extractGenAiAar) }

dependencies {
    implementation(project(":core:domain"))
    implementation(project(":core:runtime"))
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")
    implementation(files(genAiJar).builtBy(extractGenAiAar))
    testImplementation(kotlin("test"))
    testImplementation(kotlin("test-junit5"))
}

tasks.withType<Test>().configureEach { useJUnitPlatform() }
