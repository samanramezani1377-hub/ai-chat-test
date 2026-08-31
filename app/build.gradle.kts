plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.samanramezani.aichattest"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.samanramezani.aichattest"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
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
    implementation(project(":core:actions"))
    implementation(project(":core:observability"))
    implementation(project(":core:settings"))
    implementation(project(":core:runtime"))
}
