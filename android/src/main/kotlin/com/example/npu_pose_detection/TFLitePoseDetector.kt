package com.example.npu_pose_detection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.util.Log
import com.example.npu_pose_detection.models.*
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Delegate
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * TensorFlow Lite-based pose detector using MoveNet.
 *
 * Uses GPU delegate for acceleration with NNAPI/CPU fallback.
 */
class TFLitePoseDetector(private val context: Context) {

    companion object {
        private const val TAG = "TFLitePoseDetector"
        private const val MODEL_LIGHTNING = "movenet_lightning.tflite"
        private const val MODEL_THUNDER = "movenet_thunder.tflite"
        private const val INPUT_SIZE_LIGHTNING = 192
        private const val INPUT_SIZE_THUNDER = 256
    }

    private var interpreter: Interpreter? = null
    private var delegate: Delegate? = null
    private var config: DetectorConfig = DetectorConfig()
    private var _accelerationMode: AccelerationMode = AccelerationMode.UNKNOWN
    private val landmarkMapper = LandmarkMapper()

    val accelerationMode: AccelerationMode get() = _accelerationMode
    val isInitialized: Boolean get() = interpreter != null

    /**
     * Initialize the detector with configuration.
     *
     * @param config Detection configuration
     * @return The acceleration mode being used
     */
    fun initialize(config: DetectorConfig): AccelerationMode {
        this.config = config

        val modelFileName = when (config.mode) {
            DetectionMode.FAST, DetectionMode.BALANCED -> MODEL_LIGHTNING
            DetectionMode.ACCURATE -> MODEL_THUNDER
        }

        val model = loadModelFile(modelFileName)
        val delegateResult = DelegateFactory.createBestDelegate(context, config.preferredAcceleration)

        delegate = delegateResult.delegate
        _accelerationMode = delegateResult.mode

        val options = Interpreter.Options().apply {
            setNumThreads(4)
            delegateResult.delegate?.let { addDelegate(it) }
        }

        interpreter = Interpreter(model, options)
        Log.i(TAG, "Initialized with acceleration: $_accelerationMode")

        return _accelerationMode
    }

    /**
     * Update detector configuration.
     */
    fun updateConfig(config: DetectorConfig) {
        this.config = config
    }

    /**
     * Release resources.
     */
    fun dispose() {
        interpreter?.close()
        interpreter = null
        DelegateFactory.closeDelegate(delegate)
        delegate = null
        _accelerationMode = AccelerationMode.UNKNOWN
    }

    /**
     * Detect poses from image data.
     *
     * @param imageData JPEG or PNG encoded image bytes
     * @return PoseResult with detected poses
     */
    fun detectPose(imageData: ByteArray): PoseResult {
        val bitmap = BitmapFactory.decodeByteArray(imageData, 0, imageData.size)
            ?: throw IllegalArgumentException("Invalid image data")

        return detectPoseFromBitmap(bitmap)
    }

    /**
     * Detect poses from an image file.
     *
     * @param filePath Path to the image file
     * @return PoseResult with detected poses
     */
    fun detectPoseFromFile(filePath: String): PoseResult {
        val file = File(filePath)
        if (!file.exists()) {
            throw IllegalArgumentException("File not found: $filePath")
        }

        val bitmap = BitmapFactory.decodeFile(filePath)
            ?: throw IllegalArgumentException("Could not decode image: $filePath")

        return detectPoseFromBitmap(bitmap)
    }

    /**
     * Core detection from bitmap.
     */
    private fun detectPoseFromBitmap(bitmap: Bitmap): PoseResult {
        val interp = interpreter
            ?: throw IllegalStateException("Detector not initialized")

        val inputSize = getInputSize()
        val startTime = System.currentTimeMillis()

        // Resize and preprocess
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        val inputBuffer = convertBitmapToBuffer(resizedBitmap, inputSize)

        // Prepare output buffer
        // MoveNet output: [1, 1, 17, 3] for single pose
        val outputBuffer = Array(1) { Array(1) { Array(17) { FloatArray(3) } } }

        // Run inference
        interp.run(inputBuffer, outputBuffer)

        val processingTime = (System.currentTimeMillis() - startTime).toInt()

        // Map keypoints to landmarks
        val rawKeypoints = outputBuffer[0][0]
        val landmarks = landmarkMapper.mapMoveNetToPose(rawKeypoints, config.minConfidence)

        // Calculate overall score
        val detectedLandmarks = landmarks.filter { it.isDetected }
        val score = if (detectedLandmarks.isNotEmpty()) {
            detectedLandmarks.map { it.visibility }.average()
        } else {
            0.0
        }

        val pose = if (score >= config.minConfidence) {
            Pose(
                landmarks = landmarks,
                score = score,
                boundingBox = calculateBoundingBox(landmarks)
            )
        } else null

        // Clean up resized bitmap
        if (resizedBitmap != bitmap) {
            resizedBitmap.recycle()
        }

        return PoseResult(
            poses = listOfNotNull(pose),
            processingTimeMs = processingTime,
            accelerationMode = _accelerationMode,
            imageWidth = bitmap.width,
            imageHeight = bitmap.height
        )
    }

    /**
     * Get input size based on detection mode.
     */
    private fun getInputSize(): Int {
        return when (config.mode) {
            DetectionMode.FAST, DetectionMode.BALANCED -> INPUT_SIZE_LIGHTNING
            DetectionMode.ACCURATE -> INPUT_SIZE_THUNDER
        }
    }

    /**
     * Convert bitmap to TFLite input buffer.
     */
    private fun convertBitmapToBuffer(bitmap: Bitmap, inputSize: Int): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4)
        buffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        for (pixel in pixels) {
            // Normalize to [-1, 1] for MoveNet
            val r = ((pixel shr 16) and 0xFF) / 127.5f - 1.0f
            val g = ((pixel shr 8) and 0xFF) / 127.5f - 1.0f
            val b = (pixel and 0xFF) / 127.5f - 1.0f

            buffer.putFloat(r)
            buffer.putFloat(g)
            buffer.putFloat(b)
        }

        buffer.rewind()
        return buffer
    }

    /**
     * Calculate bounding box from landmarks.
     */
    private fun calculateBoundingBox(landmarks: List<PoseLandmark>): BoundingBox? {
        val detected = landmarks.filter { it.isDetected }
        if (detected.isEmpty()) return null

        val minX = detected.minOf { it.x }
        val maxX = detected.maxOf { it.x }
        val minY = detected.minOf { it.y }
        val maxY = detected.maxOf { it.y }

        // Add 10% padding
        val padding = 0.1
        val width = maxX - minX
        val height = maxY - minY

        return BoundingBox(
            left = (minX - width * padding).coerceIn(0.0, 1.0),
            top = (minY - height * padding).coerceIn(0.0, 1.0),
            width = (width * (1 + 2 * padding)).coerceIn(0.0, 1.0),
            height = (height * (1 + 2 * padding)).coerceIn(0.0, 1.0)
        )
    }

    /**
     * Process a camera frame (YUV format).
     *
     * @param planes List of plane data from camera
     * @param width Frame width
     * @param height Frame height
     * @param format Image format (yuv420, nv21, bgra8888)
     * @param rotation Rotation in degrees (0, 90, 180, 270)
     * @return PoseResult with detected poses
     */
    fun processFrame(
        planes: List<Map<String, Any>>,
        width: Int,
        height: Int,
        format: String,
        rotation: Int
    ): PoseResult {
        val bitmap = when (format) {
            "yuv420", "nv21" -> convertYuvToBitmap(planes, width, height)
            "bgra8888" -> convertBgraToBitmap(planes, width, height)
            else -> throw IllegalArgumentException("Unsupported format: $format")
        }

        val rotatedBitmap = if (rotation != 0) {
            rotateBitmap(bitmap, rotation)
        } else {
            bitmap
        }

        return detectPoseFromBitmap(rotatedBitmap).also {
            if (rotatedBitmap != bitmap) {
                rotatedBitmap.recycle()
            }
            bitmap.recycle()
        }
    }

    /**
     * Convert YUV camera frame to Bitmap.
     */
    private fun convertYuvToBitmap(planes: List<Map<String, Any>>, width: Int, height: Int): Bitmap {
        val yPlane = planes[0]["bytes"] as ByteArray
        val uPlane = if (planes.size > 1) planes[1]["bytes"] as? ByteArray else null
        val vPlane = if (planes.size > 2) planes[2]["bytes"] as? ByteArray else null

        // Combine into NV21 format for YuvImage
        val nv21: ByteArray
        if (uPlane != null && vPlane != null) {
            nv21 = ByteArray(width * height * 3 / 2)
            System.arraycopy(yPlane, 0, nv21, 0, yPlane.size)

            // Interleave U and V planes for NV21 (VU order)
            val uvSize = width * height / 4
            for (i in 0 until uvSize) {
                nv21[yPlane.size + i * 2] = vPlane[i]
                nv21[yPlane.size + i * 2 + 1] = uPlane[i]
            }
        } else {
            // Already NV21 or similar format
            nv21 = yPlane
        }

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 90, out)
        val jpegBytes = out.toByteArray()

        return BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
    }

    /**
     * Convert BGRA camera frame to Bitmap.
     */
    private fun convertBgraToBitmap(planes: List<Map<String, Any>>, width: Int, height: Int): Bitmap {
        val bytes = planes[0]["bytes"] as ByteArray
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val buffer = ByteBuffer.wrap(bytes)
        bitmap.copyPixelsFromBuffer(buffer)
        return bitmap
    }

    /**
     * Rotate bitmap by specified degrees.
     */
    private fun rotateBitmap(bitmap: Bitmap, rotation: Int): Bitmap {
        val matrix = android.graphics.Matrix()
        matrix.postRotate(rotation.toFloat())
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    /**
     * Load model file from assets.
     */
    private fun loadModelFile(modelFileName: String): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(modelFileName)
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
}
