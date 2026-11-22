import org.gradle.api.tasks.Copy

plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

val webgpuNativeAar = configurations.create("webgpuNativeAar") {
    isCanBeConsumed = false
    isCanBeResolved = true
    isTransitive = false
}
val dawnMetadataOutput = layout.buildDirectory.dir("generated/dawn-metadata")
val dawnMetadataFile = dawnMetadataOutput.map { it.file("assets/dawn_build_metadata.json") }

val extractDawnMetadata = tasks.register<Copy>("extractDawnMetadata") {
    from(zipTree(webgpuNativeAar.singleFile)) {
        include("assets/dawn_build_metadata.json")
        includeEmptyDirs = false
    }
    into(dawnMetadataOutput)
}

android {
    namespace = "com.woogit.aicore.runtime.android"
    compileSdk = 36
    ndkVersion = "27.2.12479018"

    defaultConfig {
        minSdk = 29
        ndk {
            abiFilters += "arm64-v8a"
        }
        externalNativeBuild {
            cmake {
                arguments += listOf(
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DAI_CHAT_DAWN_METADATA_FILE=${dawnMetadataFile.get().asFile.absolutePath}"
                )
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildTypes {
        debug {
            isMinifyEnabled = false
        }
        release {
            isMinifyEnabled = false
        }
    }
}

dependencies {
    add(webgpuNativeAar.name, "androidx.webgpu:webgpu:1.0.0-alpha06@aar")
}

tasks.configureEach {
    if (name.startsWith("configureCMake")) {
        dependsOn(extractDawnMetadata)
    }
}
