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

## Performance

| Device | Chipset | Acceleration | Inference Time |
|--------|---------|--------------|----------------|
| Galaxy S24 Ultra | Snapdragon 8 Elite | GPU | ~15ms |
| iPhone 15 Pro | A17 Pro | ANE (CoreML) | ~12ms |
| Pixel 8 | Tensor G3 | GPU | ~18ms |

## Platform Support

| Platform | ML Framework | Model | Acceleration |
|----------|-------------|-------|--------------|
| iOS 14+ | TFLite + CoreML | MediaPipe Pose | Neural Engine → GPU → CPU |
| Android API 24+ | MediaPipe Tasks | PoseLandmarker Lite | GPU → CPU |

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
        minSdkVersion 24
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
final result = await detector.initialize();
print('Running on: ${result.accelerationMode}'); // GPU or CPU

// Detect pose from image
final imageBytes = await File('image.jpg').readAsBytes();
final poseResult = await detector.detectPose(imageBytes);

print('Inference time: ${poseResult.processingTimeMs}ms');

if (poseResult.hasPoses) {
  final pose = poseResult.firstPose!;
  print('Detected ${pose.landmarks.length} landmarks');

  // Access specific landmarks (MediaPipe 33-point format)
  final nose = pose.getLandmark(MediaPipeLandmarkType.nose);
  final leftShoulder = pose.getLandmark(MediaPipeLandmarkType.leftShoulder);
  print('Nose at (${nose.x}, ${nose.y})');
}

// Clean up
detector.dispose();
```

## MediaPipe 33 Landmarks

```
0: nose
1-6: eyes (inner, center, outer)
7-8: ears
9-10: mouth corners
11-12: shoulders
13-14: elbows
15-16: wrists
17-22: hands (pinky, index, thumb)
23-24: hips
25-26: knees
27-28: ankles
29-30: heels
31-32: foot index
```

## Model Architecture

This plugin uses **MediaPipe PoseLandmarker** (2-stage pipeline):

| Stage | Model | Input Size | Output |
|-------|-------|------------|--------|
| 1. Person Detection | pose_detector | 224x224 | Bounding box |
| 2. Landmark Detection | pose_landmarks_detector | 256x256 | 33 landmarks (x, y, z, visibility) |

## Documentation

- [API Reference](https://pub.dev/documentation/flutter_pose_detection/latest/)
- [CHANGELOG](CHANGELOG.md)

## License

MIT License - see [LICENSE](LICENSE)
