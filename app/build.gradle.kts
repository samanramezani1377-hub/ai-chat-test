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
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"))
        }
    }

    packagingOptions {
        jniLibs {
            // Keep native libraries uncompressed and avoid a second extracted copy on disk.
            useLegacyPackaging = false
        }
    }
}

dependencies {
    implementation(project(":core:domain"))
    implementation(project(":core:actions"))
    implementation(project(":core:observability"))
    implementation(project(":core:settings"))
    implementation(project(":core:runtime"))
    implementation(project(":core:runtime-android"))
    implementation(project(":core:conversation"))
    implementation(project(":core:agent"))

    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")

    // Current stable Compose BOM: keeps the complete UI stack on a compatible modern release.
    implementation(platform("androidx.compose:compose-bom:2025.02.00"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.foundation:foundation-layout")
    implementation("androidx.compose.material3:material3:1.3.2")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.compose.ui:ui-tooling-preview")
    debugImplementation("androidx.compose.ui:ui-tooling")
    debugImplementation("androidx.compose.ui:ui-test-manifest")

    implementation("androidx.activity:activity-compose:1.10.1")
    implementation("androidx.core:core-ktx:1.17.0")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    implementation("androidx.compose.material3:material3-adaptive-navigation-suite")

    androidTestImplementation(platform("androidx.compose:compose-bom:2026.09.00"))
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")
}
