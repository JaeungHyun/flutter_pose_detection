package com.example.npu_pose_detection

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.util.Log
import com.example.npu_pose_detection.models.PoseResult
import kotlinx.coroutines.*
import java.io.File

/**
 * Video processor for frame-by-frame pose analysis.
 */
class VideoProcessor(
    private val context: Context,
    private val detector: LiteRtPoseDetector
) {
    companion object {
        private const val TAG = "VideoProcessor"
    }

    private var isCancelled = false
    private var job: Job? = null
    private var progressCallback: ((VideoAnalysisProgress) -> Unit)? = null

    /**
     * Analyze a video file for poses.
     *
     * @param videoPath Path to the video file
     * @param frameInterval Process every Nth frame (1 = all frames)
     * @param progress Progress callback
     * @return VideoAnalysisResult with all detected poses
     */
    suspend fun analyzeVideo(
        videoPath: String,
        frameInterval: Int = 1,
        progress: (VideoAnalysisProgress) -> Unit
    ): VideoAnalysisResult = withContext(Dispatchers.IO) {
        progressCallback = progress
        isCancelled = false

        val file = File(videoPath)
        if (!file.exists()) {
            throw VideoProcessorException("File not found: $videoPath")
        }

        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(videoPath)

            // Get video properties
            val durationMs = retriever.extractMetadata(
                MediaMetadataRetriever.METADATA_KEY_DURATION
            )?.toLongOrNull() ?: 0L

            val durationSeconds = durationMs / 1000.0

            val width = retriever.extractMetadata(
                MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH
            )?.toIntOrNull() ?: 0

            val height = retriever.extractMetadata(
                MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT
            )?.toIntOrNull() ?: 0

            val frameRateString = retriever.extractMetadata(
                MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE
            )
            val frameRate = frameRateString?.toDoubleOrNull() ?: 30.0

            val totalFrames = (durationSeconds * frameRate).toInt()

            val frameResults = mutableListOf<VideoFrameResult>()
            var analyzedCount = 0
            val startTime = System.currentTimeMillis()

            // Calculate frame timestamps to extract
            val frameTimestampsUs = mutableListOf<Long>()
            var frameIndex = 0
            while (frameIndex < totalFrames) {
                if (isCancelled) break

                if (frameIndex % frameInterval == 0) {
                    val timestampUs = ((frameIndex.toDouble() / frameRate) * 1_000_000).toLong()
                    frameTimestampsUs.add(timestampUs)
                }
                frameIndex++
            }

            // Process each frame
            for ((index, timestampUs) in frameTimestampsUs.withIndex()) {
                if (isCancelled) break

                val actualFrameIndex = index * frameInterval
                val timestampSeconds = timestampUs / 1_000_000.0

                try {
                    // Extract frame at timestamp
                    val bitmap = retriever.getFrameAtTime(
                        timestampUs,
                        MediaMetadataRetriever.OPTION_CLOSEST
                    )

                    if (bitmap != null) {
                        // Process the frame
                        val poseResult = detector.detectPose(bitmapToByteArray(bitmap))
                        bitmap.recycle()

                        frameResults.add(
                            VideoFrameResult(
                                frameIndex = actualFrameIndex,
                                timestampSeconds = timestampSeconds,
                                result = poseResult
                            )
                        )
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to process frame at ${timestampSeconds}s: ${e.message}")
                    // Continue with next frame
                }

                analyzedCount++

                // Report progress
                val elapsed = System.currentTimeMillis() - startTime
                val framesPerSecond = if (elapsed > 0) analyzedCount * 1000.0 / elapsed else 0.0
                val remainingFrames = frameTimestampsUs.size - index - 1
                val estimatedRemaining = if (framesPerSecond > 0) {
                    remainingFrames / framesPerSecond
                } else null

                withContext(Dispatchers.Main) {
                    progress(
                        VideoAnalysisProgress(
                            currentFrame = actualFrameIndex,
                            totalFrames = totalFrames,
                            currentTimeSeconds = timestampSeconds,
                            durationSeconds = durationSeconds,
                            estimatedRemainingSeconds = estimatedRemaining
                        )
                    )
                }
            }

            val totalTime = (System.currentTimeMillis() - startTime).toInt()

            VideoAnalysisResult(
                frames = frameResults,
                totalFrames = totalFrames,
                analyzedFrames = analyzedCount,
                durationSeconds = durationSeconds,
                frameRate = frameRate,
                width = width,
                height = height,
                totalAnalysisTimeMs = totalTime
            )
        } finally {
            retriever.release()
        }
    }

    /**
     * Cancel ongoing analysis.
     */
    fun cancel() {
        isCancelled = true
        job?.cancel()
    }

    private fun bitmapToByteArray(bitmap: Bitmap): ByteArray {
        val stream = java.io.ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 90, stream)
        return stream.toByteArray()
    }
}

/**
 * Exception thrown during video processing.
 */
class VideoProcessorException(message: String) : Exception(message)

/**
 * Progress during video analysis.
 */
data class VideoAnalysisProgress(
    val currentFrame: Int,
    val totalFrames: Int,
    val currentTimeSeconds: Double,
    val durationSeconds: Double,
    val estimatedRemainingSeconds: Double?
) {
    fun toMap(): Map<String, Any?> = mapOf(
        "type" to "progress",
        "currentFrame" to currentFrame,
        "totalFrames" to totalFrames,
        "currentTimeSeconds" to currentTimeSeconds,
        "durationSeconds" to durationSeconds,
        "estimatedRemainingSeconds" to estimatedRemainingSeconds
    )
}

/**
 * Result for a single video frame.
 */
data class VideoFrameResult(
    val frameIndex: Int,
    val timestampSeconds: Double,
    val result: PoseResult
) {
    fun toMap(): Map<String, Any> = mapOf(
        "frameIndex" to frameIndex,
        "timestampSeconds" to timestampSeconds,
        "result" to result.toMap()
    )
}

/**
 * Result of video analysis.
 */
data class VideoAnalysisResult(
    val frames: List<VideoFrameResult>,
    val totalFrames: Int,
    val analyzedFrames: Int,
    val durationSeconds: Double,
    val frameRate: Double,
    val width: Int,
    val height: Int,
    val totalAnalysisTimeMs: Int
) {
    fun toMap(): Map<String, Any> = mapOf(
        "frames" to frames.map { it.toMap() },
        "totalFrames" to totalFrames,
        "analyzedFrames" to analyzedFrames,
        "durationSeconds" to durationSeconds,
        "frameRate" to frameRate,
        "width" to width,
        "height" to height,
        "totalAnalysisTimeMs" to totalAnalysisTimeMs
    )
}
