# LiteRT 2.1.0 (successor to TensorFlow Lite)
# https://ai.google.dev/edge/litert/android
-keep class com.google.ai.edge.litert.** { *; }
-keep class com.google.ai.edge.litert.gpu.** { *; }
-keep class com.google.ai.edge.litert.Accelerator { *; }
-keep class com.google.ai.edge.litert.CompiledModel { *; }
-keep class com.google.ai.edge.litert.CompiledModel$Options { *; }
-keep class com.google.ai.edge.litert.TensorBuffer { *; }
-dontwarn com.google.ai.edge.litert.**

# Keep TFLite classes for iOS compatibility
-keep class org.tensorflow.lite.** { *; }
-dontwarn org.tensorflow.lite.**
