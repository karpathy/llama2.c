import org.jetbrains.kotlin.de.undercouch.gradle.tasks.download.Download

@Suppress("DSL_SCOPE_VIOLATION") // TODO: Remove once KTIJ-19369 is fixed
plugins {
    alias(libs.plugins.androidApplication)
    alias(libs.plugins.kotlinAndroid)
    alias(libs.plugins.download)
}

android {
    namespace = "com.celikin.llama2"
    compileSdk = 33

    defaultConfig {
        applicationId = "com.celikin.llama2"
        minSdk = 24
        targetSdk = 33
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

    sourceSets {
        maybeCreate("main").apply {
            assets {
                srcDirs("src/main/assets")
            }
        }
    }

    kotlinOptions {
        jvmTarget = "1.8"
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    buildFeatures {
        viewBinding = true
    }
}


tasks {
    val downloadTokenizer by creating(Download::class) {
        onlyIf { !file("$projectDir/src/main/assets/tokenizer.bin").exists() }
        src("https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin")
        dest("$projectDir/src/main/assets/tokenizer.bin")
    }
    val downloadModel by creating(Download::class) {
        onlyIf { !file("$projectDir/src/main/assets/stories15M.bin").exists() }
        src("https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin")
        dest("$projectDir/src/main/assets/stories15M.bin")
    }
    whenTaskAdded {
        if (name in listOf("assembleDebug", "assembleRelease")) {
            dependsOn(downloadTokenizer)
            dependsOn(downloadModel)
        }
    }
}

dependencies {
    implementation(libs.androidx.activity.ktx)
    implementation(libs.androidx.lifecycle.viewmodel.ktx)
    implementation(libs.androidx.lifecycle.livedata.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)

    implementation(libs.coroutines.core)
    implementation(libs.core.ktx)
    implementation(libs.appcompat)
    implementation(libs.material)
    implementation(libs.constraintlayout)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.test.ext.junit)
    androidTestImplementation(libs.espresso.core)
}