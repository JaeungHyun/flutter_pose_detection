import Foundation

/// Maps MoveNet 17 landmarks to MediaPipe 33 landmarks.
///
/// MoveNet provides 17 COCO keypoints, but we output 33 landmarks for
/// MediaPipe/ML Kit compatibility. Missing landmarks are set to visibility 0.
class LandmarkMapper {

    /// MoveNet index to MediaPipe index mapping.
    ///
    /// MoveNet keypoints (COCO format):
    /// 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
    /// 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
    /// 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
    /// 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
    private let moveNetToMediaPipe: [Int: Int] = [
        0: 0,   // nose -> nose
        1: 2,   // left_eye -> leftEye
        2: 5,   // right_eye -> rightEye
        3: 7,   // left_ear -> leftEar
        4: 8,   // right_ear -> rightEar
        5: 11,  // left_shoulder -> leftShoulder
        6: 12,  // right_shoulder -> rightShoulder
        7: 13,  // left_elbow -> leftElbow
        8: 14,  // right_elbow -> rightElbow
        9: 15,  // left_wrist -> leftWrist
        10: 16, // right_wrist -> rightWrist
        11: 23, // left_hip -> leftHip
        12: 24, // right_hip -> rightHip
        13: 25, // left_knee -> leftKnee
        14: 26, // right_knee -> rightKnee
        15: 27, // left_ankle -> leftAnkle
        16: 28  // right_ankle -> rightAnkle
    ]

    /// Map MoveNet keypoints to 33-landmark MediaPipe format.
    ///
    /// - Parameters:
    ///   - keypoints: Array of 17 keypoints, each [y, x, confidence]
    ///   - minConfidence: Minimum confidence threshold
    /// - Returns: Array of 33 PoseLandmarks
    func mapMoveNetToPose(
        _ keypoints: [[Float]],
        minConfidence: Float
    ) -> [PoseLandmark] {
        var landmarks = (0..<LandmarkType.count).map { index in
            PoseLandmark.notDetected(type: LandmarkType(rawValue: index)!)
        }

        // Map MoveNet keypoints to MediaPipe indices
        for (moveNetIndex, point) in keypoints.enumerated() {
            guard let mediaPipeIndex = moveNetToMediaPipe[moveNetIndex] else { continue }

            // MoveNet output format: [y, x, confidence]
            let y = Double(point[0])
            let x = Double(point[1])
            let confidence = Double(point[2])

            if confidence >= Double(minConfidence) {
                landmarks[mediaPipeIndex] = PoseLandmark(
                    type: LandmarkType(rawValue: mediaPipeIndex)!,
                    x: x,
                    y: y,
                    z: 0.0,
                    visibility: confidence
                )
            }
        }

        // Interpolate derived landmarks
        interpolateHeels(&landmarks)

        return landmarks
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
