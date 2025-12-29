/// Hardware-accelerated pose detection using native ML frameworks.
///
/// This library provides pose detection capabilities using:
/// - iOS: Apple Vision Framework with Neural Engine acceleration
/// - Android: TensorFlow Lite with GPU/NNAPI/CPU delegates
///
/// ## Quick Start
///
/// ```dart
/// import 'package:flutter_pose_detection/flutter_pose_detection.dart';
///
/// // Create and initialize detector
/// final detector = NpuPoseDetector();
/// await detector.initialize();
///
/// // Detect pose from image
/// final result = await detector.detectPose(imageBytes);
/// if (result.hasPoses) {
///   final pose = result.firstPose!;
///   final nose = pose.getLandmark(LandmarkType.nose);
///   print('Nose at (${nose.x}, ${nose.y})');
/// }
///
/// // Clean up
/// detector.dispose();
/// ```
///
/// ## Features
///
/// - **33 landmarks** in MediaPipe-compatible format
/// - **Hardware acceleration** (NPU/GPU) when available
/// - **Image detection** for static images
/// - **Camera stream** for realtime detection
/// - **Video analysis** for recorded files
///
/// See also:
/// - [NpuPoseDetector] - Main detector class
/// - [Pose] - Detected pose with landmarks
/// - [PoseLandmark] - Individual body landmark
/// - [LandmarkType] - 33 landmark types
library;

// Models
export 'src/models/acceleration_mode.dart';
export 'src/models/bounding_box.dart';
export 'src/models/detection_mode.dart';
export 'src/models/detector_config.dart';
export 'src/models/frame_result.dart';
export 'src/models/landmark_type.dart';
export 'src/models/pose.dart';
export 'src/models/pose_landmark.dart';
export 'src/models/pose_result.dart';
export 'src/models/video_analysis_result.dart';

// Errors
export 'src/errors/detection_error.dart';

// Main detector
export 'src/pose_detector.dart';

// Platform interface (for platform implementers)
export 'src/platform/pose_detector_platform.dart'
    show PoseDetectorPlatform, PoseDetectorChannels;
