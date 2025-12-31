# Flutter Pose Detection

Hardware-accelerated pose detection Flutter plugin using LiteRT (Google's successor to TensorFlow Lite) with native NPU acceleration.

[![pub package](https://img.shields.io/pub/v/flutter_pose_detection.svg)](https://pub.dev/packages/flutter_pose_detection)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **True NPU Acceleration**: QNN Delegate (Snapdragon HTP), Core ML (Apple ANE)
- **Cross-Platform**: iOS (Core ML) and Android (LiteRT + QNN)
- **17 COCO Landmarks**: Industry-standard body pose format
- **Ultra-Fast**: 6-12ms on flagship NPUs, 100+ FPS capable
- **Automatic Fallback**: NPU → GPU → CPU graceful degradation

## Performance

| Device | Chipset | Acceleration | Inference Time |
|--------|---------|--------------|----------------|
| Galaxy S24 Ultra | Snapdragon 8 Elite | NPU (Hexagon HTP) | **6-12ms** |
| Galaxy S24 | Snapdragon 8 Gen 3 | NPU (Hexagon HTP) | ~8ms |
| iPhone 15 Pro | A17 Pro | ANE (Core ML) | ~10ms |
| Pixel 8 | Tensor G3 | GPU Delegate | ~15ms |

## Platform Support

| Platform | ML Framework | Model | Acceleration |
|----------|-------------|-------|--------------|
| iOS 15+ | Core ML | HRNetPose | Neural Engine (ANE) |
| Android API 31+ | LiteRT + QNN SDK 2.41 | HRNetPose w8a8 | Hexagon HTP (NPU) |

### Supported Chipsets

**Android (NPU via QNN Delegate):**
- Qualcomm Snapdragon 8 Elite, 8 Gen 3/2/1, 7+ Gen 2

**Android (GPU Fallback):**
- Samsung Exynos 2400/2200
- Google Tensor G3/G2
- MediaTek Dimensity 9300/9200

**iOS:**
- A12 Bionic and newer (Neural Engine)

## Installation

```yaml
dependencies:
  flutter_pose_detection: ^0.2.1
```

### iOS Setup

Update `ios/Podfile`:
```ruby
platform :ios, '15.0'
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
        ndk {
            abiFilters 'arm64-v8a'  // NPU only supports arm64
        }
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
final mode = await detector.initialize();
print('Running on: $mode'); // NPU, GPU, or CPU

// Detect pose from image
final imageBytes = await File('image.jpg').readAsBytes();
final result = await detector.detectPose(imageBytes);

print('Inference time: ${result.processingTimeMs}ms');
print('Acceleration: ${result.accelerationMode}');

if (result.hasPoses) {
  final pose = result.firstPose!;
  print('Detected ${pose.landmarks.length} landmarks');

  // Access specific landmarks (COCO 17-point format)
  final nose = pose.getLandmark(LandmarkType.nose);
  final leftShoulder = pose.getLandmark(LandmarkType.leftShoulder);
  print('Nose at (${nose.x}, ${nose.y})');
}

// Clean up
detector.dispose();
```

## COCO 17 Keypoints

```
0: nose
1: left_eye        2: right_eye
3: left_ear        4: right_ear
5: left_shoulder   6: right_shoulder
7: left_elbow      8: right_elbow
9: left_wrist      10: right_wrist
11: left_hip       12: right_hip
13: left_knee      14: right_knee
15: left_ankle     16: right_ankle
```

## Model Architecture

This plugin uses **HRNet-W32** (High-Resolution Network) optimized for mobile inference:

| Property | Value |
|----------|-------|
| Input Size | 192x256 (WxH) |
| Output | 17x48x64 heatmaps |
| Quantization | INT8 (w8a8) for NPU |
| Model Size | ~28MB (TFLite), ~55MB (Core ML) |

## Documentation

- [API Reference](https://pub.dev/documentation/flutter_pose_detection/latest/)
- [Quickstart Guide](doc/quickstart.md)
- [Landmark Reference](doc/landmarks.md)

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

## License

MIT License - see [LICENSE](LICENSE)
