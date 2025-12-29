# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-12-30

### Added
- Initial release of NPU Pose Detection plugin
- **iOS Support**: Apple Vision Framework with Neural Engine acceleration
  - Automatic NPU/GPU acceleration on A12+ devices
  - VNDetectHumanBodyPoseRequest for 17 body landmarks
  - Landmark mapping to MediaPipe 33-point format
- **Android Support**: TensorFlow Lite with MoveNet Lightning model
  - GPU Delegate for hardware acceleration
  - NNAPI support for devices API 27-34
  - CPU fallback with XNNPack optimization
- **Core Features**:
  - Static image pose detection (`detectPose`, `detectPoseFromFile`)
  - Real-time camera frame processing (`processFrame`, `startCameraDetection`)
  - Video file analysis (`analyzeVideo` with progress tracking)
  - Configurable detection parameters (`PoseDetectorConfig`)
- **Data Models**:
  - `Pose` with 33 MediaPipe-compatible landmarks
  - `PoseLandmark` with normalized coordinates (0-1) and confidence scores
  - `BoundingBox` for detected person region
  - `LandmarkType` enum for easy landmark access
- **Error Handling**:
  - Typed `DetectionError` with error codes
  - Graceful fallback from NPU to GPU to CPU
- **Example App**:
  - Image detection demo with gallery picker
  - Real-time camera detection with pose overlay
  - Video analysis with progress UI
