-keep class com.k2fsa.sherpa.onnx.** { *; }
-keepclasseswithmembernames class com.k2fsa.sherpa.onnx.** { native <methods>; }

# Rust JNI symbols use the exact Kotlin class and native method names.
-keep class com.samanramezani.aichattest.voice.LayaNativeTokenizer { *; }
