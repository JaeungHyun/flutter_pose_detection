package com.example.npu_pose_detection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.util.Log
import com.example.npu_pose_detection.models.*
import com.qualcomm.qti.QnnDelegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * TFLite-based pose detector with hardware acceleration.
 *
 * Supports two inference paths:
 * - NPU: TFLite Interpreter + QnnDelegate for HRNetPose (1.3-1.6ms on S25)
 * - CPU: TFLite Interpreter for MoveNet fallback (3-9ms)
 *
 * Uses Qualcomm QNN SDK for true NPU/HTP acceleration on Snapdragon devices.
 */
class LiteRtPoseDetector(private val context: Context) {

    companion object {
        private const val TAG = "LiteRtPoseDetector"

        // HRNetPose - Qualcomm optimized for NPU (w8a8 quantized)
        private const val MODEL_HRNETPOSE = "hrnetpose_w8a8.tflite"
        private const val HRNET_INPUT_WIDTH = 192
        private const val HRNET_INPUT_HEIGHT = 256
        private const val HRNET_HEATMAP_WIDTH = 48
        private const val HRNET_HEATMAP_HEIGHT = 64

        // MoveNet - CPU fallback models
        private const val MODEL_MOVENET_LIGHTNING = "movenet_lightning.tflite"
        private const val MODEL_MOVENET_THUNDER = "movenet_thunder.tflite"
        private const val MOVENET_LIGHTNING_SIZE = 192
        private const val MOVENET_THUNDER_SIZE = 256

        private const val NUM_KEYPOINTS = 17
    }

    // TFLite Interpreter (used for NPU, GPU, and CPU)
    private var interpreter: Interpreter? = null
    private var qnnDelegate: QnnDelegate? = null
    private var gpuDelegate: GpuDelegate? = null

    private var config: DetectorConfig = DetectorConfig()
    private var _accelerationMode: AccelerationMode = AccelerationMode.UNKNOWN
    private var useHRNetPose: Boolean = false
    private val landmarkMapper = LandmarkMapper()

    val accelerationMode: AccelerationMode get() = _accelerationMode
    val isInitialized: Boolean get() = interpreter != null

    /**
     * Initialize the detector with configuration.
     *
     * Chipset-aware initialization based on the architecture report:
     * - Snapdragon: QNN Delegate (Hexagon NPU) → GPU Delegate → CPU
     * - Exynos/Tensor/MediaTek: GPU Delegate (skip unstable NNAPI) → CPU
     * - Others: GPU Delegate → CPU
     *
     * @param config Detection configuration
     * @return The acceleration mode being used
     */
    fun initialize(config: DetectorConfig): AccelerationMode {
        this.config = config
        val nativeLibDir = context.applicationInfo.nativeLibraryDir

        // Detect chipset for optimal delegate selection
        val chipset = ChipsetDetector.detectChipset()
        Log.i(TAG, "Device info:\n${ChipsetDetector.getDeviceInfo()}")

        // Strategy based on chipset type (per report recommendations)
        when (chipset) {
            ChipsetDetector.ChipsetType.QUALCOMM_SNAPDRAGON -> {
                // Snapdragon: Try QNN Delegate first for best NPU performance
                if (config.preferredAcceleration != AccelerationMode.CPU) {
                    if (tryInitializeWithQNN(nativeLibDir)) {
                        return _accelerationMode
                    }
                    // Fallback to GPU if QNN fails
                    if (tryInitializeWithGPU()) {
                        return _accelerationMode
                    }
                }
            }
            ChipsetDetector.ChipsetType.SAMSUNG_EXYNOS,
            ChipsetDetector.ChipsetType.GOOGLE_TENSOR,
            ChipsetDetector.ChipsetType.MEDIATEK -> {
                // Non-Snapdragon: Skip NNAPI (deprecated), use GPU directly
                if (config.preferredAcceleration != AccelerationMode.CPU) {
                    if (tryInitializeWithGPU()) {
                        return _accelerationMode
                    }
                }
            }
            ChipsetDetector.ChipsetType.OTHER -> {
                // Unknown chipset: Try GPU with fallback to CPU
                if (config.preferredAcceleration != AccelerationMode.CPU) {
                    if (tryInitializeWithGPU()) {
                        return _accelerationMode
                    }
                }
            }
        }

        // Final fallback: MoveNet on CPU with XNNPACK
        return initializeCPUFallback()
    }

    /**
     * Try to initialize with QNN Delegate (Snapdragon only).
     * Uses Hexagon Tensor Processor (HTP) for NPU acceleration.
     */
    private fun tryInitializeWithQNN(nativeLibDir: String): Boolean {
        try {
            val modelFile = copyModelToCache(MODEL_HRNETPOSE)
            Log.i(TAG, "Trying HRNetPose with QNN Delegate (Hexagon HTP)...")

            // Configure QNN Delegate for HTP backend
            // Based on Qualcomm AI Engine Direct SDK recommendations
            val qnnOptions = QnnDelegate.Options().apply {
                setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND)
                setSkelLibraryDir(nativeLibDir)
                // Additional options for performance
                // setPerformanceMode(QnnDelegate.Options.PerformanceMode.HIGH_PERFORMANCE)
            }
            Log.d(TAG, "QNN config: HTP_BACKEND, skelLibDir=$nativeLibDir")

            qnnDelegate = QnnDelegate(qnnOptions)

            val interpreterOptions = Interpreter.Options().apply {
                addDelegate(qnnDelegate)
                setNumThreads(4)
            }

            interpreter = Interpreter(loadModelFile(modelFile), interpreterOptions)
            useHRNetPose = true
            _accelerationMode = AccelerationMode.NPU
            Log.i(TAG, "✓ Initialized HRNetPose with QNN NPU (Hexagon HTP)")
            return true
        } catch (e: Throwable) {
            // Catch Throwable to handle NoSuchMethodError from native JNI layer
            Log.w(TAG, "✗ QNN NPU initialization failed: ${e.javaClass.simpleName}: ${e.message}")
            cleanup()
            return false
        }
    }

    /**
     * Try to initialize with GPU Delegate (OpenCL/OpenGL).
     * Works on all Android devices with GPU support.
     */
    private fun tryInitializeWithGPU(): Boolean {
        try {
            val modelFile = copyModelToCache(MODEL_HRNETPOSE)
            Log.i(TAG, "Trying HRNetPose with GPU Delegate (OpenCL)...")

            // GPU Delegate uses OpenCL for Mali, Adreno, and other GPUs
            gpuDelegate = GpuDelegate()

            val interpreterOptions = Interpreter.Options().apply {
                addDelegate(gpuDelegate)
                setNumThreads(4)
            }

            interpreter = Interpreter(loadModelFile(modelFile), interpreterOptions)
            useHRNetPose = true
            _accelerationMode = AccelerationMode.GPU
            Log.i(TAG, "✓ Initialized HRNetPose with GPU Delegate")
            return true
        } catch (e: Exception) {
            Log.w(TAG, "✗ GPU Delegate initialization failed: ${e.message}")
            cleanup()
            return false
        }
    }

    /**
     * Clean up delegates and interpreter on initialization failure.
     */
    private fun cleanup() {
        try {
            interpreter?.close()
            qnnDelegate?.close()
            gpuDelegate?.close()
        } catch (e: Exception) {
            Log.w(TAG, "Error during cleanup: ${e.message}")
        }
        interpreter = null
        qnnDelegate = null
        gpuDelegate = null
    }

    /**
     * Initialize CPU fallback with XNNPACK optimization.
     */
    private fun initializeCPUFallback(): AccelerationMode {
        // Fallback: MoveNet on CPU
        try {
            val modelName = getMoveNetModelFileName()
            val modelFile = copyModelToCache(modelName)
            Log.i(TAG, "Fallback to MoveNet CPU: $modelName")

            val interpreterOptions = Interpreter.Options().apply {
                setNumThreads(4)
            }

            interpreter = Interpreter(loadModelFile(modelFile), interpreterOptions)
            useHRNetPose = false
            _accelerationMode = AccelerationMode.CPU
            Log.i(TAG, "Initialized MoveNet with CPU")
            return _accelerationMode
        } catch (e: Exception) {
            Log.e(TAG, "All initialization attempts failed: ${e.message}")
            throw e
        }
    }

    /**
     * Load model file as MappedByteBuffer for TFLite Interpreter.
     */
    private fun loadModelFile(file: File): MappedByteBuffer {
        FileInputStream(file).use { inputStream ->
            val fileChannel = inputStream.channel
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, fileChannel.size())
        }
    }

    /**
     * Get MoveNet model filename based on detection mode.
     */
    private fun getMoveNetModelFileName(): String {
        return when (config.mode) {
            DetectionMode.FAST, DetectionMode.BALANCED -> MODEL_MOVENET_LIGHTNING
            DetectionMode.ACCURATE -> MODEL_MOVENET_THUNDER
        }
    }

    /**
     * Copy model file from assets to cache directory.
     */
    private fun copyModelToCache(modelFileName: String): File {
        val cacheFile = File(context.cacheDir, modelFileName)
        if (!cacheFile.exists()) {
            context.assets.open(modelFileName).use { input ->
                cacheFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            Log.d(TAG, "Copied model to cache: $modelFileName")
        }
        return cacheFile
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
        try {
            interpreter?.close()
            qnnDelegate?.close()
            gpuDelegate?.close()
        } catch (e: Exception) {
            Log.w(TAG, "Error closing resources: ${e.message}")
        }
        interpreter = null
        qnnDelegate = null
        gpuDelegate = null
        useHRNetPose = false
        _accelerationMode = AccelerationMode.UNKNOWN
        Log.d(TAG, "Detector disposed")
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
     * Uses TFLite Interpreter with QnnDelegate for NPU, or plain Interpreter for CPU.
     */
    private fun detectPoseFromBitmap(bitmap: Bitmap): PoseResult {
        if (!isInitialized) {
            throw IllegalStateException("Detector not initialized")
        }

        val startTime = System.currentTimeMillis()

        // Get input dimensions based on model
        val (inputWidth, inputHeight) = if (useHRNetPose) {
            HRNET_INPUT_WIDTH to HRNET_INPUT_HEIGHT
        } else {
            val size = getMoveNetInputSize()
            size to size
        }

        Log.d(TAG, "detectPoseFromBitmap: model=${if (useHRNetPose) "HRNetPose" else "MoveNet"}, " +
                   "mode=$_accelerationMode, input=${inputWidth}x${inputHeight}")

        // Resize to model input size
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true)

        val keypoints: Array<FloatArray> = if (useHRNetPose) {
            runHRNetPoseInference(resizedBitmap, inputWidth, inputHeight)
        } else {
            runMoveNetInference(resizedBitmap, inputWidth, inputHeight)
        }

        val processingTime = (System.currentTimeMillis() - startTime).toInt()

        val landmarks = landmarkMapper.mapMoveNetToPose(keypoints, config.minConfidence)

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

        val result = PoseResult(
            poses = listOfNotNull(pose),
            processingTimeMs = processingTime,
            accelerationMode = _accelerationMode,
            imageWidth = bitmap.width,
            imageHeight = bitmap.height
        )
        Log.d(TAG, "Detection result: ${result.poses.size} poses, ${processingTime}ms, ${result.accelerationMode}")
        return result
    }

    /**
     * Run HRNetPose inference (NPU path via QnnDelegate).
     * Input: [1, 256, 192, 3] as UINT8
     * Output: [1, 17, 64, 48] heatmaps
     */
    private fun runHRNetPoseInference(bitmap: Bitmap, width: Int, height: Int): Array<FloatArray> {
        val interp = interpreter!!

        // Prepare input: [1, height, width, 3] as UINT8
        val inputBuffer = ByteBuffer.allocateDirect(1 * height * width * 3)
        inputBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        for (pixel in pixels) {
            inputBuffer.put(((pixel shr 16) and 0xFF).toByte())  // R
            inputBuffer.put(((pixel shr 8) and 0xFF).toByte())   // G
            inputBuffer.put((pixel and 0xFF).toByte())           // B
        }
        inputBuffer.rewind()

        // Prepare output: HRNetPose outputs heatmaps [1, 17, 64, 48]
        val outputBuffer = ByteBuffer.allocateDirect(1 * NUM_KEYPOINTS * HRNET_HEATMAP_HEIGHT * HRNET_HEATMAP_WIDTH * 4)
        outputBuffer.order(ByteOrder.nativeOrder())

        // Run inference
        interp.run(inputBuffer, outputBuffer)

        // Parse heatmaps to keypoints
        outputBuffer.rewind()
        val floatBuffer = outputBuffer.asFloatBuffer()
        val heatmapSize = HRNET_HEATMAP_WIDTH * HRNET_HEATMAP_HEIGHT

        return Array(NUM_KEYPOINTS) { k ->
            var maxVal = Float.MIN_VALUE
            var maxIdx = 0

            for (i in 0 until heatmapSize) {
                val value = floatBuffer.get(k * heatmapSize + i)
                if (value > maxVal) {
                    maxVal = value
                    maxIdx = i
                }
            }

            val heatmapX = maxIdx % HRNET_HEATMAP_WIDTH
            val heatmapY = maxIdx / HRNET_HEATMAP_WIDTH
            val normalizedX = heatmapX.toFloat() / HRNET_HEATMAP_WIDTH
            val normalizedY = heatmapY.toFloat() / HRNET_HEATMAP_HEIGHT
            val confidence = 1.0f / (1.0f + kotlin.math.exp(-maxVal))

            floatArrayOf(normalizedY, normalizedX, confidence)
        }
    }

    /**
     * Run MoveNet inference (CPU path).
     * Input: [1, size, size, 3] as INT32 (0-255)
     * Output: [1, 1, 17, 3] keypoints (y, x, confidence)
     */
    private fun runMoveNetInference(bitmap: Bitmap, width: Int, height: Int): Array<FloatArray> {
        val interp = interpreter!!

        // Prepare input: [1, height, width, 3] as INT32
        val inputBuffer = ByteBuffer.allocateDirect(1 * height * width * 3 * 4)
        inputBuffer.order(ByteOrder.nativeOrder())
        val intBuffer = inputBuffer.asIntBuffer()

        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        for (pixel in pixels) {
            intBuffer.put((pixel shr 16) and 0xFF)  // R
            intBuffer.put((pixel shr 8) and 0xFF)   // G
            intBuffer.put(pixel and 0xFF)           // B
        }
        inputBuffer.rewind()

        // Prepare output: [1, 1, 17, 3]
        val outputBuffer = ByteBuffer.allocateDirect(1 * 1 * NUM_KEYPOINTS * 3 * 4)
        outputBuffer.order(ByteOrder.nativeOrder())

        // Run inference
        interp.run(inputBuffer, outputBuffer)

        // Parse output to keypoints
        outputBuffer.rewind()
        val floatBuffer = outputBuffer.asFloatBuffer()

        return Array(NUM_KEYPOINTS) { i ->
            floatArrayOf(
                floatBuffer.get(i * 3),     // y
                floatBuffer.get(i * 3 + 1), // x
                floatBuffer.get(i * 3 + 2)  // confidence
            )
        }
    }

    /**
     * Get MoveNet input size based on detection mode.
     */
    private fun getMoveNetInputSize(): Int {
        return when (config.mode) {
            DetectionMode.FAST, DetectionMode.BALANCED -> MOVENET_LIGHTNING_SIZE
            DetectionMode.ACCURATE -> MOVENET_THUNDER_SIZE
        }
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
}
