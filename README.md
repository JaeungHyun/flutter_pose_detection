# Flutter Pose Detection

Hardware-accelerated pose detection Flutter plugin using MediaPipe PoseLandmarker with GPU acceleration.

[![pub package](https://img.shields.io/pub/v/flutter_pose_detection.svg)](https://pub.dev/packages/flutter_pose_detection)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **MediaPipe PoseLandmarker**: Official Google pose detection API
- **33 Landmarks**: Full body tracking including hands and feet
- **Cross-Platform**: iOS (CoreML/Metal) and Android (GPU Delegate)
- **Hardware Acceleration**: Automatic GPU fallback to CPU
- **Real-time**: Camera frame processing with FPS tracking
- **Video Analysis**: Process video files with progress tracking
- **Angle Calculation**: Built-in utilities for body angle measurements

## Performance

| Device | Chipset | Acceleration | Inference Time |
|--------|---------|--------------|----------------|
| Galaxy S25 Ultra | Snapdragon 8 Elite | GPU | <25ms |

## Platform Support

| Platform | ML Framework | Model | Acceleration |
|----------|-------------|-------|--------------|
| iOS 14+ | TFLite 2.14 + CoreML/Metal | pose_detector + pose_landmarks_detector | Neural Engine → GPU → CPU |
| Android API 31+ | MediaPipe Tasks 0.10.14 | pose_landmarker_lite.task | GPU → CPU |

## Installation

```yaml
dependencies:
  flutter_pose_detection: ^0.3.0
```

### iOS Setup

Update `ios/Podfile`:
```ruby
platform :ios, '14.0'
```

Add camera permission to `ios/Runner/Info.plist`:
```xml
<key>NSCameraUsageDescription</key>
<string>Camera access is needed for pose detection</string>
```

### Android Setup

Update `android/app/build.gradle`:
```gradle
android {
    defaultConfig {
        minSdkVersion 31
    }
}
```

Add camera permission to `android/app/src/main/AndroidManifest.xml`:
```xml
<uses-permission android:name="android.permission.CAMERA" />
```

## Quick Start

```dart
import 'package:flutter_pose_detection/flutter_pose_detection.dart';

// Create and initialize detector
final detector = NpuPoseDetector();
final accelerationMode = await detector.initialize();
print('Running on: $accelerationMode'); // gpu, npu, or cpu

// Detect pose from image
final imageBytes = await File('image.jpg').readAsBytes();
final result = await detector.detectPose(imageBytes);

print('Inference time: ${result.processingTimeMs}ms');

if (result.hasPoses) {
  final pose = result.firstPose!;
  print('Detected ${pose.landmarks.length} landmarks');

  // Access specific landmarks (MediaPipe 33-point format)
  final nose = pose.getLandmark(LandmarkType.nose);
  final leftShoulder = pose.getLandmark(LandmarkType.leftShoulder);
  print('Nose at (${nose.x}, ${nose.y})');
}

// Clean up
detector.dispose();
```

## Configuration

```dart
// Default configuration
final detector = NpuPoseDetector();

// Realtime camera optimization (fast mode, low latency)
final detector = NpuPoseDetector(
  config: PoseDetectorConfig.realtime(),
);

// Accuracy optimization (for still images)
final detector = NpuPoseDetector(
  config: PoseDetectorConfig.accurate(),
);

// Custom configuration
final detector = NpuPoseDetector(
  config: PoseDetectorConfig(
    mode: DetectionMode.accurate,
    maxPoses: 3,
    minConfidence: 0.6,
    enableZEstimation: true,
  ),
);
```

## Camera Stream Processing

```dart
// Process camera frames manually
cameraController.startImageStream((CameraImage image) async {
  final planes = image.planes.map((p) => {
    'bytes': p.bytes,
    'bytesPerRow': p.bytesPerRow,
    'bytesPerPixel': p.bytesPerPixel,
  }).toList();

  final result = await detector.processFrame(
    planes: planes,
    width: image.width,
    height: image.height,
    format: 'yuv420',
    rotation: 90,
  );

  if (result.hasPoses) {
    // Draw skeleton overlay
  }
});

// Or use built-in camera detection stream
final subscription = detector.startCameraDetection().listen(
  (frameResult) {
    print('FPS: ${frameResult.fps.toStringAsFixed(1)}');
    if (frameResult.result.hasPoses) {
      // Process pose
    }
  },
);

// Stop when done
await detector.stopCameraDetection();
```

## Video Analysis

```dart
// Subscribe to progress updates
detector.videoAnalysisProgress.listen((progress) {
  print('${(progress.progress * 100).toStringAsFixed(1)}% complete');
});

// Analyze video file
final result = await detector.analyzeVideo(
  '/path/to/video.mp4',
  frameInterval: 3, // Analyze every 3rd frame
);

print('Analyzed ${result.analyzedFrames} frames');
print('Detection rate: ${(result.detectionRate * 100).toStringAsFixed(0)}%');

// Access individual frame results
for (final frame in result.frames) {
  if (frame.result.hasPoses) {
    print('Pose at ${frame.timestampSeconds}s');
  }
}
```

## Body Angle Calculation

```dart
if (result.hasPoses) {
  final pose = result.firstPose!;

  // Calculate knee angle
  final kneeAngle = pose.calculateAngle(
    LandmarkType.leftHip,
    LandmarkType.leftKnee,
    LandmarkType.leftAnkle,
  );

  if (kneeAngle != null) {
    print('Knee angle: ${kneeAngle.toStringAsFixed(1)}°');
  }

  // Calculate shoulder width
  final shoulderWidth = pose.calculateDistance(
    LandmarkType.leftShoulder,
    LandmarkType.rightShoulder,
  );

  // Get visible landmarks only
  final visibleLandmarks = pose.getVisibleLandmarks(threshold: 0.5);
  print('${visibleLandmarks.length} landmarks visible');
}
```

## MediaPipe 33 Landmarks

| Index | Name | Description |
|-------|------|-------------|
| 0 | nose | Nose tip |
| 1-3 | leftEyeInner, leftEye, leftEyeOuter | Left eye |
| 4-6 | rightEyeInner, rightEye, rightEyeOuter | Right eye |
| 7-8 | leftEar, rightEar | Ears |
| 9-10 | mouthLeft, mouthRight | Mouth corners |
| 11-12 | leftShoulder, rightShoulder | Shoulders |
| 13-14 | leftElbow, rightElbow | Elbows |
| 15-16 | leftWrist, rightWrist | Wrists |
| 17-22 | pinky, index, thumb (L/R) | Hand landmarks |
| 23-24 | leftHip, rightHip | Hips |
| 25-26 | leftKnee, rightKnee | Knees |
| 27-28 | leftAnkle, rightAnkle | Ankles |
| 29-30 | leftHeel, rightHeel | Heels |
| 31-32 | leftFootIndex, rightFootIndex | Foot tips |

## Model Architecture

### iOS (TFLite 2-stage pipeline)

| Stage | Model | Input Size | Output |
|-------|-------|------------|--------|
| 1. Person Detection | pose_detector.tflite | 224x224 | Bounding box |
| 2. Landmark Detection | pose_landmarks_detector.tflite | 256x256 | 33 landmarks (x, y, z, visibility) |

### Android (MediaPipe Tasks)

| Model | Input Size | Output |
|-------|------------|--------|
| pose_landmarker_lite.task | 256x256 | 33 landmarks (x, y, z, visibility, presence) |

## API Reference

### NpuPoseDetector

| Method | Description |
|--------|-------------|
| `initialize()` | Load ML model, returns `AccelerationMode` |
| `detectPose(Uint8List)` | Detect pose from image bytes |
| `detectPoseFromFile(String)` | Detect pose from file path |
| `processFrame(...)` | Process single camera frame |
| `startCameraDetection()` | Start camera stream detection |
| `stopCameraDetection()` | Stop camera detection |
| `analyzeVideo(String)` | Analyze video file |
| `cancelVideoAnalysis()` | Cancel ongoing video analysis |
| `updateConfig(PoseDetectorConfig)` | Update configuration |
| `dispose()` | Release resources |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `isInitialized` | `bool` | Whether detector is ready |
| `accelerationMode` | `AccelerationMode` | Current hardware acceleration |
| `config` | `PoseDetectorConfig` | Current configuration |
| `videoAnalysisProgress` | `Stream<VideoAnalysisProgress>` | Video analysis progress |

## Documentation

- [API Reference](https://pub.dev/documentation/flutter_pose_detection/latest/)
- [CHANGELOG](CHANGELOG.md)

## License

MIT License - see [LICENSE](LICENSE)
