pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        maven { url = uri("https://jitpack.io") }
    }
}

rootProject.name = "AIChatTest"
include(":app")
include(":core:domain")
include(":core:actions")
include(":core:observability")
include(":core:settings")
include(":core:runtime")
include(":core:conversation")
include(":core:agent")
include(":core:runtime-android")
