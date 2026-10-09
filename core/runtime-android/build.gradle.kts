plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.woogit.aicore.runtime.android"
    compileSdk = 36
    ndkVersion = "27.2.12479018"

    defaultConfig {
        minSdk = 29
        consumerProguardFiles("consumer-rules.pro")
        ndk {
            debugSymbolLevel = "none"
        }
        externalNativeBuild {
            cmake {
                // Validation is intentionally debug-only while native loading is under investigation.
                // Keep the native build strictly single-ABI; the app also packages arm64-v8a only.
                abiFilters += "arm64-v8a"
                arguments += listOf(
                    "-DLLAMA_BUILD_COMMON=OFF",
                    "-DLLAMA_BUILD_TESTS=OFF",
                    "-DLLAMA_BUILD_EXAMPLES=OFF",
                    "-DLLAMA_BUILD_TOOLS=OFF",
                    "-DLLAMA_BUILD_SERVER=OFF",
                    "-DLLAMA_BUILD_APP=OFF",
                    "-DLLAMA_BUILD_UI=OFF",
                    "-DLLAMA_OPENSSL=OFF",
                    "-DGGML_NATIVE=OFF",
                    "-DGGML_CPU_ARM_ARCH=armv8-a",
                    "-DGGML_OPENMP=OFF",
                    "-DGGML_LLAMAFILE=OFF",
                    "-DGGML_OPENCL=OFF",
                    "-DGGML_VULKAN=ON",
                    "-DGGML_VULKAN_RUN_TESTS=OFF",
                    "-DGGML_VULKAN_VALIDATE=OFF",
                    "-DGGML_VULKAN=OFF",
                    "-DCMAKE_BUILD_TYPE=Release",
                )
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
        }
    }

    buildTypes {
        debug {
            externalNativeBuild {
                cmake {
                    arguments += "-DGGML_VULKAN=ON"
                }
            }
        }
        release {
            externalNativeBuild {
                cmake {
                    // Keep the real OpenCL kernel profile available in the shipped
                    // diagnostic build; the profiler writes asynchronously to app cache.
                    arguments += "-DGGML_VULKAN=ON"
                }
            }
        }
    }


    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation(project(":core:domain"))
    implementation(project(":core:runtime"))
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")
    testImplementation(kotlin("test"))
    testImplementation(kotlin("test-junit5"))
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
