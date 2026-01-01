package com.example.npu_pose_detection.models

import com.example.npu_pose_detection.MediaPipeLandmarkType

/**
 * Landmark type following COCO 17-keypoint topology.
 * Same as MoveNet/HRNet output format.
 *
 * @deprecated Use MediaPipeLandmarkType for 33-landmark format
 */
enum class LandmarkType(val index: Int) {
    NOSE(0),
    LEFT_EYE(1),
    RIGHT_EYE(2),
    LEFT_EAR(3),
    RIGHT_EAR(4),
    LEFT_SHOULDER(5),
    RIGHT_SHOULDER(6),
    LEFT_ELBOW(7),
    RIGHT_ELBOW(8),
    LEFT_WRIST(9),
    RIGHT_WRIST(10),
    LEFT_HIP(11),
    RIGHT_HIP(12),
    LEFT_KNEE(13),
    RIGHT_KNEE(14),
    LEFT_ANKLE(15),
    RIGHT_ANKLE(16);

    companion object {
        const val COUNT = 17

        fun fromIndex(index: Int): LandmarkType {
            return values().firstOrNull { it.index == index }
                ?: throw IllegalArgumentException("Invalid landmark index: $index")
        }
    }
}

/**
 * Individual body landmark supporting both COCO (17) and MediaPipe (33) formats.
 *
 * For MediaPipe 33-landmark format, use the constructor with MediaPipeLandmarkType.
 * For COCO 17-landmark format, use the constructor with LandmarkType.
 */
data class PoseLandmark(
    val typeIndex: Int,
    val x: Double,
    val y: Double,
    val z: Double = 0.0,
    val visibility: Double
) {
    val isDetected: Boolean get() = visibility > 0

    // Legacy constructor for COCO LandmarkType
    constructor(
        type: LandmarkType,
        x: Double,
        y: Double,
        z: Double = 0.0,
        visibility: Double
    ) : this(type.index, x, y, z, visibility)

    // Constructor for MediaPipe 33-landmark type
    constructor(
        type: MediaPipeLandmarkType,
        x: Double,
        y: Double,
        z: Double = 0.0,
        visibility: Double
    ) : this(type.index, x, y, z, visibility)

    companion object {
        fun notDetected(type: LandmarkType): PoseLandmark {
            return PoseLandmark(type.index, 0.0, 0.0, 0.0, 0.0)
        }

        fun notDetected(type: MediaPipeLandmarkType): PoseLandmark {
            return PoseLandmark(type.index, 0.0, 0.0, 0.0, 0.0)
        }
    }

    fun toMap(): Map<String, Any> {
        return mapOf(
            "type" to typeIndex,
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
