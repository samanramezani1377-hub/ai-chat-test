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
    }
}

rootProject.name = "AIChatTest"
include(":app")
include(":core:domain")
include(":core:actions")
include(":core:observability")
include(":core:settings")
include(":core:runtime")
include(":core:runtime-android")
include(":core:conversation")
include(":core:agent")
include(":llama-kt")
project(":llama-kt").projectDir = file("libs/llama.kt/llama-kt")
