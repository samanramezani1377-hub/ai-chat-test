-keep class com.k2fsa.sherpa.onnx.** { *; }
-keepclasseswithmembernames class com.k2fsa.sherpa.onnx.** { native <methods>; }

# Needle's native JNI entry points are resolved by their exact class and method names.
-keep class com.samanramezani.aichattest.voice.CactusNeedleNative { *; }
