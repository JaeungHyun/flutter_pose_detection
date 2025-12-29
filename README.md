# Flutter Pose Detection

Hardware-accelerated pose detection Flutter plugin using native ML frameworks.

[![pub package](https://img.shields.io/pub/v/flutter_pose_detection.svg)](https://pub.dev/packages/flutter_pose_detection)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Hardware Acceleration**: Automatic NPU/GPU acceleration on supported devices
- **Cross-Platform**: iOS (Vision Framework) and Android (TensorFlow Lite)
- **33 Landmarks**: MediaPipe-compatible body pose format
- **Multiple Modes**: Image, camera stream, and video file analysis
- **High Performance**: 15+ FPS realtime, <50ms single image

## Platform Support

| Platform | ML Framework | Acceleration |
|----------|-------------|--------------|
| iOS 14+  | Vision Framework | Neural Engine (automatic) |
| Android API 26+ | TensorFlow Lite | GPU Delegate / NNAPI / CPU |

## Installation

```yaml
dependencies:
  flutter_pose_detection: ^0.1.0
```

### iOS Setup

Add to `ios/Podfile`:
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
        minSdkVersion 26
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
await detector.initialize();

// Detect pose from image
final imageBytes = await File('image.jpg').readAsBytes();
final result = await detector.detectPose(imageBytes);

if (result.hasPoses) {
  final pose = result.firstPose!;
  print('Detected ${pose.landmarks.length} landmarks');

  // Access specific landmarks
  final nose = pose.getLandmark(LandmarkType.nose);
  print('Nose at (${nose.x}, ${nose.y})');
}

// Clean up
detector.dispose();
```

## Documentation

- [API Reference](https://pub.dev/documentation/flutter_pose_detection/latest/)
- [Quickstart Guide](doc/quickstart.md)
- [Landmark Reference](doc/landmarks.md)

## License

MIT License - see [LICENSE](LICENSE)
