package com.example.npu_pose_detection

import com.example.npu_pose_detection.models.*

/**
 * Maps MoveNet 17 landmarks to MediaPipe 33 landmarks.
 *
 * MoveNet provides 17 keypoints, but we output 33 landmarks for
 * MediaPipe/ML Kit compatibility. Missing landmarks have visibility 0.
 */
class LandmarkMapper {

    /**
     * MoveNet index to MediaPipe index mapping.
     *
     * MoveNet keypoints:
     * 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
     * 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
     * 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
     * 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
     */
    private val moveNetToMediaPipe = mapOf(
        0 to 0,   // nose -> nose
        1 to 2,   // left_eye -> leftEye
        2 to 5,   // right_eye -> rightEye
        3 to 7,   // left_ear -> leftEar
        4 to 8,   // right_ear -> rightEar
        5 to 11,  // left_shoulder -> leftShoulder
        6 to 12,  // right_shoulder -> rightShoulder
        7 to 13,  // left_elbow -> leftElbow
        8 to 14,  // right_elbow -> rightElbow
        9 to 15,  // left_wrist -> leftWrist
        10 to 16, // right_wrist -> rightWrist
        11 to 23, // left_hip -> leftHip
        12 to 24, // right_hip -> rightHip
        13 to 25, // left_knee -> leftKnee
        14 to 26, // right_knee -> rightKnee
        15 to 27, // left_ankle -> leftAnkle
        16 to 28  // right_ankle -> rightAnkle
    )

    /**
     * Map MoveNet keypoints to 33-landmark MediaPipe format.
     *
     * @param keypoints Array of 17 keypoints, each [y, x, confidence]
     * @param minConfidence Minimum confidence threshold
     * @return List of 33 PoseLandmark
     */
    fun mapMoveNetToPose(
        keypoints: Array<FloatArray>,
        minConfidence: Float
    ): List<PoseLandmark> {
        val landmarks = MutableList(LandmarkType.COUNT) { index ->
            PoseLandmark.notDetected(LandmarkType.fromIndex(index))
        }

        // Map MoveNet keypoints to MediaPipe indices
        keypoints.forEachIndexed { moveNetIndex, point ->
            val mediaPipeIndex = moveNetToMediaPipe[moveNetIndex] ?: return@forEachIndexed

            // MoveNet output format: [y, x, confidence]
            val y = point[0]
            val x = point[1]
            val confidence = point[2]

            if (confidence >= minConfidence) {
                landmarks[mediaPipeIndex] = PoseLandmark(
                    type = LandmarkType.fromIndex(mediaPipeIndex),
                    x = x.toDouble(),
                    y = y.toDouble(),
                    z = 0.0,
                    visibility = confidence.toDouble()
                )
            }
        }

        // Interpolate derived landmarks
        interpolateHeels(landmarks)

        return landmarks
    }

    /**
     * Interpolate heel positions from ankle and knee.
     */
    private fun interpolateHeels(landmarks: MutableList<PoseLandmark>) {
        // Left heel (index 29) from left knee (25) and left ankle (27)
        val leftKnee = landmarks[25]
        val leftAnkle = landmarks[27]
        if (leftKnee.isDetected && leftAnkle.isDetected) {
            landmarks[29] = interpolatePoint(leftKnee, leftAnkle, 29)
        }

        // Right heel (index 30) from right knee (26) and right ankle (28)
        val rightKnee = landmarks[26]
        val rightAnkle = landmarks[28]
        if (rightKnee.isDetected && rightAnkle.isDetected) {
            landmarks[30] = interpolatePoint(rightKnee, rightAnkle, 30)
        }
    }

    /**
     * Interpolate a point by extending the line from `from` through `through`.
     */
    private fun interpolatePoint(
        from: PoseLandmark,
        through: PoseLandmark,
        index: Int
    ): PoseLandmark {
        val dx = through.x - from.x
        val dy = through.y - from.y
        val extensionFactor = 0.15  // Extend 15% beyond ankle

        return PoseLandmark(
            type = LandmarkType.fromIndex(index),
            x = through.x + dx * extensionFactor,
            y = through.y + dy * extensionFactor,
            z = through.z,
            visibility = minOf(from.visibility, through.visibility) * 0.8
        )
    }
}
