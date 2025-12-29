import Foundation
import Vision

/// Maps Vision Framework 19 landmarks to MediaPipe 33 landmarks.
///
/// Vision provides 19 body pose landmarks, but we need to output 33 landmarks
/// for MediaPipe/ML Kit compatibility. Missing landmarks are set to visibility 0.
class LandmarkMapper {

    /// Vision joint name to MediaPipe index mapping.
    private let visionToMediaPipe: [VNHumanBodyPoseObservation.JointName: Int] = [
        .nose: 0,
        .leftEye: 2,
        .rightEye: 5,
        .leftEar: 7,
        .rightEar: 8,
        .leftShoulder: 11,
        .rightShoulder: 12,
        .leftElbow: 13,
        .rightElbow: 14,
        .leftWrist: 15,
        .rightWrist: 16,
        .leftHip: 23,
        .rightHip: 24,
        .leftKnee: 25,
        .rightKnee: 26,
        .leftAnkle: 27,
        .rightAnkle: 28,
    ]

    /// Map a Vision body pose observation to our 33-landmark format.
    ///
    /// - Parameter observation: The Vision framework observation
    /// - Returns: A Pose with 33 landmarks
    func mapVisionToPose(_ observation: VNHumanBodyPoseObservation) -> Pose {
        var landmarks = (0..<LandmarkType.count).map { index in
            PoseLandmark.notDetected(type: LandmarkType(rawValue: index)!)
        }

        // Map direct landmarks from Vision
        for (jointName, mediaPipeIndex) in visionToMediaPipe {
            if let point = try? observation.recognizedPoint(jointName),
               point.confidence > 0 {
                landmarks[mediaPipeIndex] = PoseLandmark(
                    type: LandmarkType(rawValue: mediaPipeIndex)!,
                    x: Double(point.location.x),
                    y: Double(1.0 - point.location.y),  // Vision uses bottom-left origin, flip Y
                    z: 0.0,  // Vision doesn't provide depth
                    visibility: Double(point.confidence)
                )
            }
        }

        // Interpolate heel positions from ankle + knee
        interpolateHeels(&landmarks)

        return Pose(
            landmarks: landmarks,
            score: Double(observation.confidence),
            boundingBox: nil  // Vision doesn't provide bounding box for body pose
        )
    }

    /// Interpolate heel positions based on ankle and knee positions.
    ///
    /// Heels are estimated by extending the ankle-knee line past the ankle.
    private func interpolateHeels(_ landmarks: inout [PoseLandmark]) {
        // Left heel (index 29)
        if landmarks[25].isDetected && landmarks[27].isDetected {
            let knee = landmarks[25]  // leftKnee
            let ankle = landmarks[27]  // leftAnkle
            landmarks[29] = interpolatePoint(from: knee, through: ankle, index: 29)
        }

        // Right heel (index 30)
        if landmarks[26].isDetected && landmarks[28].isDetected {
            let knee = landmarks[26]  // rightKnee
            let ankle = landmarks[28]  // rightAnkle
            landmarks[30] = interpolatePoint(from: knee, through: ankle, index: 30)
        }
    }

    /// Interpolate a point by extending the line from `from` through `through`.
    ///
    /// - Parameters:
    ///   - from: Starting point (e.g., knee)
    ///   - through: Point to extend through (e.g., ankle)
    ///   - index: MediaPipe landmark index for the result
    /// - Returns: Interpolated landmark
    private func interpolatePoint(
        from: PoseLandmark,
        through: PoseLandmark,
        index: Int
    ) -> PoseLandmark {
        let dx = through.x - from.x
        let dy = through.y - from.y
        let extensionFactor = 0.15  // Extend 15% beyond the ankle

        return PoseLandmark(
            type: LandmarkType(rawValue: index)!,
            x: through.x + dx * extensionFactor,
            y: through.y + dy * extensionFactor,
            z: through.z,
            visibility: min(from.visibility, through.visibility) * 0.8  // Lower confidence for interpolated
        )
    }
}
