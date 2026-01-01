package com.example.npu_pose_detection

import com.example.npu_pose_detection.models.*

/**
 * Maps MoveNet/HRNet 17 keypoints to COCO 17 format directly.
 *
 * COCO 17 keypoints (same as MoveNet/HRNet output):
 * 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
 * 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
 * 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
 * 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
 */
class LandmarkMapper {

    companion object {
        const val NUM_KEYPOINTS = 17
    }

    /**
     * Map keypoints to COCO 17 format (direct 1:1 mapping).
     *
     * @param keypoints Array of 17 keypoints, each [y, x, confidence]
     * @param minConfidence Minimum confidence threshold
     * @return List of 17 PoseLandmark in COCO format
     */
    fun mapMoveNetToPose(
        keypoints: Array<FloatArray>,
        minConfidence: Float
    ): List<PoseLandmark> {
        return keypoints.mapIndexed { index, point ->
            // Output format: [y, x, confidence]
            val y = point[0]
            val x = point[1]
            val confidence = point[2]

            if (confidence >= minConfidence) {
                PoseLandmark(
                    type = LandmarkType.fromIndex(index),
                    x = x.toDouble(),
                    y = y.toDouble(),
                    z = 0.0,
                    visibility = confidence.toDouble()
                )
            } else {
                PoseLandmark.notDetected(LandmarkType.fromIndex(index))
            }
        }
    }
}
