plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}
android {
    namespace = "com.woogit.aicore.runtime.mnn"
    compileSdk = 36
    ndkVersion = "27.2.12479018"
    defaultConfig {
        minSdk = 29
        ndk { abiFilters += "arm64-v8a" }
        externalNativeBuild { cmake { arguments += listOf("-DMNN_BUILD_LLM=ON", "-DMNN_OPENCL=ON", "-DMNN_ARM82=ON", "-DMNN_LOW_MEMORY=ON", "-DMNN_SUPPORT_TRANSFORMER_FUSE=ON", "-DMNN_BUILD_TEST=OFF", "-DMNN_BUILD_DEMO=OFF", "-DCMAKE_BUILD_TYPE=Release") } }
    }
    externalNativeBuild { cmake { path = file("src/main/cpp/CMakeLists.txt") } }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    kotlinOptions { jvmTarget = "17" }
}
dependencies {
    implementation(project(":core:domain"))
    implementation(project(":core:runtime"))
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")
    testImplementation(kotlin("test"))
    testImplementation(kotlin("test-junit5"))
}
tasks.withType<Test>().configureEach { useJUnitPlatform() }
