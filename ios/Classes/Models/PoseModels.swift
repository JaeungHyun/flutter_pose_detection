import Foundation

/// Landmark type following MediaPipe 33-landmark topology.
enum LandmarkType: Int, CaseIterable {
    case nose = 0
    case leftEyeInner = 1
    case leftEye = 2
    case leftEyeOuter = 3
    case rightEyeInner = 4
    case rightEye = 5
    case rightEyeOuter = 6
    case leftEar = 7
    case rightEar = 8
    case mouthLeft = 9
    case mouthRight = 10
    case leftShoulder = 11
    case rightShoulder = 12
    case leftElbow = 13
    case rightElbow = 14
    case leftWrist = 15
    case rightWrist = 16
    case leftPinky = 17
    case rightPinky = 18
    case leftIndex = 19
    case rightIndex = 20
    case leftThumb = 21
    case rightThumb = 22
    case leftHip = 23
    case rightHip = 24
    case leftKnee = 25
    case rightKnee = 26
    case leftAnkle = 27
    case rightAnkle = 28
    case leftHeel = 29
    case rightHeel = 30
    case leftFootIndex = 31
    case rightFootIndex = 32

    static let count = 33
}

/// Individual body landmark.
struct PoseLandmark {
    let type: LandmarkType
    let x: Double
    let y: Double
    let z: Double
    let visibility: Double

    var isDetected: Bool { visibility > 0 }

    static func notDetected(type: LandmarkType) -> PoseLandmark {
        return PoseLandmark(type: type, x: 0, y: 0, z: 0, visibility: 0)
    }

    func toDictionary() -> [String: Any] {
        return [
            "type": type.rawValue,
            "x": x,
            "y": y,
            "z": z,
            "visibility": visibility
        ]
    }
}

/// Bounding box for detected pose.
struct BoundingBox {
    let left: Double
    let top: Double
    let width: Double
    let height: Double

    func toDictionary() -> [String: Any] {
        return [
            "left": left,
            "top": top,
            "width": width,
            "height": height
        ]
    }
}

/// A single detected pose with 33 landmarks.
struct Pose {
    let landmarks: [PoseLandmark]
    let score: Double
    let boundingBox: BoundingBox?

    func toDictionary() -> [String: Any] {
        var dict: [String: Any] = [
            "landmarks": landmarks.map { $0.toDictionary() },
            "score": score
        ]
        if let box = boundingBox {
            dict["boundingBox"] = box.toDictionary()
        }
        return dict
    }
}

/// Result from pose detection.
struct PoseResult {
    let poses: [Pose]
    let processingTimeMs: Int
    let accelerationMode: AccelerationMode
    let timestamp: Date
    let imageWidth: Int
    let imageHeight: Int

    static func empty(
        processingTimeMs: Int,
        imageWidth: Int,
        imageHeight: Int
    ) -> PoseResult {
        return PoseResult(
            poses: [],
            processingTimeMs: processingTimeMs,
            accelerationMode: .npu,
            timestamp: Date(),
            imageWidth: imageWidth,
            imageHeight: imageHeight
        )
    }

    func toDictionary() -> [String: Any] {
        return [
            "poses": poses.map { $0.toDictionary() },
            "processingTimeMs": processingTimeMs,
            "accelerationMode": accelerationMode.rawValue,
            "timestamp": Int(timestamp.timeIntervalSince1970 * 1000),
            "imageWidth": imageWidth,
            "imageHeight": imageHeight
        ]
    }
}
