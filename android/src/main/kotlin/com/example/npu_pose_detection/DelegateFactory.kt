package com.example.npu_pose_detection

import android.content.Context
import android.os.Build
import android.util.Log
import com.example.npu_pose_detection.models.AccelerationMode
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.gpu.GpuDelegate

/**
 * Factory for creating TensorFlow Lite delegates.
 *
 * Tries delegates in order: GPU -> NNAPI -> CPU (XNNPack)
 */
object DelegateFactory {

    private const val TAG = "DelegateFactory"

    /**
     * Result of delegate creation.
     */
    data class DelegateResult(
        val delegate: Delegate?,
        val mode: AccelerationMode
    )

    /**
     * Create the best available delegate for this device.
     *
     * @param context Android context
     * @param preferred Optional preferred acceleration mode
     * @return DelegateResult with delegate and mode
     */
    fun createBestDelegate(
        context: Context,
        preferred: AccelerationMode?
    ): DelegateResult {
        // If CPU is explicitly requested, skip hardware delegates
        if (preferred == AccelerationMode.CPU) {
            return DelegateResult(null, AccelerationMode.CPU)
        }

        // Try GPU delegate first (most reliable)
        if (preferred == null || preferred == AccelerationMode.GPU) {
            val gpuResult = tryGpuDelegate()
            if (gpuResult != null) {
                return gpuResult
            }
        }

        // Try NNAPI for older devices (API 27-34)
        if (preferred == null || preferred == AccelerationMode.NPU) {
            if (isNnapiSupported()) {
                val nnapiResult = tryNnapiDelegate()
                if (nnapiResult != null) {
                    return nnapiResult
                }
            }
        }

        // Fallback to CPU with XNNPack
        Log.i(TAG, "Using CPU (XNNPack) delegate")
        return DelegateResult(null, AccelerationMode.CPU)
    }

    /**
     * Try to create a GPU delegate.
     */
    private fun tryGpuDelegate(): DelegateResult? {
        return try {
            val delegate = GpuDelegate()
            Log.i(TAG, "GPU delegate created successfully")
            DelegateResult(delegate, AccelerationMode.GPU)
        } catch (e: Exception) {
            Log.w(TAG, "GPU delegate unavailable: ${e.message}")
            null
        }
    }

    /**
     * Try to create an NNAPI delegate.
     *
     * Note: NNAPI is deprecated in Android 15 (API 35).
     */
    private fun tryNnapiDelegate(): DelegateResult? {
        if (!isNnapiSupported()) return null

        return try {
            // NNAPI delegate creation would go here
            // For now, we skip NNAPI and prefer GPU
            Log.i(TAG, "NNAPI available but skipping in favor of GPU")
            null
        } catch (e: Exception) {
            Log.w(TAG, "NNAPI delegate unavailable: ${e.message}")
            null
        }
    }

    /**
     * Check if NNAPI is supported on this device.
     *
     * NNAPI is available on API 27+ but deprecated in API 35.
     */
    private fun isNnapiSupported(): Boolean {
        return Build.VERSION.SDK_INT in 27..34
    }

    /**
     * Close a delegate if it's not null.
     */
    fun closeDelegate(delegate: Delegate?) {
        try {
            when (delegate) {
                is GpuDelegate -> delegate.close()
                // Add other delegate types as needed
            }
        } catch (e: Exception) {
            Log.w(TAG, "Error closing delegate: ${e.message}")
        }
    }
}
