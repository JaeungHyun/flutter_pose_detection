package com.example.npu_pose_detection.models

/**
 * Landmark type following MediaPipe 33-landmark topology.
 */
enum class LandmarkType(val index: Int) {
    NOSE(0),
    LEFT_EYE_INNER(1),
    LEFT_EYE(2),
    LEFT_EYE_OUTER(3),
    RIGHT_EYE_INNER(4),
    RIGHT_EYE(5),
    RIGHT_EYE_OUTER(6),
    LEFT_EAR(7),
    RIGHT_EAR(8),
    MOUTH_LEFT(9),
    MOUTH_RIGHT(10),
    LEFT_SHOULDER(11),
    RIGHT_SHOULDER(12),
    LEFT_ELBOW(13),
    RIGHT_ELBOW(14),
    LEFT_WRIST(15),
    RIGHT_WRIST(16),
    LEFT_PINKY(17),
    RIGHT_PINKY(18),
    LEFT_INDEX(19),
    RIGHT_INDEX(20),
    LEFT_THUMB(21),
    RIGHT_THUMB(22),
    LEFT_HIP(23),
    RIGHT_HIP(24),
    LEFT_KNEE(25),
    RIGHT_KNEE(26),
    LEFT_ANKLE(27),
    RIGHT_ANKLE(28),
    LEFT_HEEL(29),
    RIGHT_HEEL(30),
    LEFT_FOOT_INDEX(31),
    RIGHT_FOOT_INDEX(32);

    companion object {
        const val COUNT = 33

        fun fromIndex(index: Int): LandmarkType {
            return values().firstOrNull { it.index == index }
                ?: throw IllegalArgumentException("Invalid landmark index: $index")
        }
    }
}

/**
 * Individual body landmark.
 */
data class PoseLandmark(
    val type: LandmarkType,
    val x: Double,
    val y: Double,
    val z: Double = 0.0,
    val visibility: Double
) {
    val isDetected: Boolean get() = visibility > 0

    companion object {
        fun notDetected(type: LandmarkType): PoseLandmark {
            return PoseLandmark(type, 0.0, 0.0, 0.0, 0.0)
        }
    }

    fun toMap(): Map<String, Any> {
        return mapOf(
            "type" to type.index,
            "x" to x,
            "y" to y,
            "z" to z,
            "visibility" to visibility
        )
    }
}

/**
 * Bounding box for detected pose.
 */
data class BoundingBox(
    val left: Double,
    val top: Double,
    val width: Double,
    val height: Double
) {
    fun toMap(): Map<String, Any> {
        return mapOf(
            "left" to left,
            "top" to top,
            "width" to width,
            "height" to height
        )
    }
}

/**
 * A single detected pose with 33 landmarks.
 */
data class Pose(
    val landmarks: List<PoseLandmark>,
    val score: Double,
    val boundingBox: BoundingBox? = null
) {
    fun toMap(): Map<String, Any?> {
        return mapOf(
            "landmarks" to landmarks.map { it.toMap() },
            "score" to score,
            "boundingBox" to boundingBox?.toMap()
        )
    }
}

/**
 * Result from pose detection.
 */
data class PoseResult(
    val poses: List<Pose>,
    val processingTimeMs: Int,
    val accelerationMode: AccelerationMode,
    val timestamp: Long = System.currentTimeMillis(),
    val imageWidth: Int,
    val imageHeight: Int
) {
    companion object {
        fun empty(
            processingTimeMs: Int,
            accelerationMode: AccelerationMode,
            imageWidth: Int,
            imageHeight: Int
        ): PoseResult {
            return PoseResult(
                poses = emptyList(),
                processingTimeMs = processingTimeMs,
                accelerationMode = accelerationMode,
                imageWidth = imageWidth,
                imageHeight = imageHeight
            )
        }
    }

    fun toMap(): Map<String, Any> {
        return mapOf(
            "poses" to poses.map { it.toMap() },
            "processingTimeMs" to processingTimeMs,
            "accelerationMode" to accelerationMode.name.lowercase(),
            "timestamp" to timestamp,
            "imageWidth" to imageWidth,
            "imageHeight" to imageHeight
        )
    }
}
