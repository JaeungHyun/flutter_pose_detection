package com.example.npu_pose_detection

import android.os.Build
import android.util.Log

/**
 * Chipset detection utility for optimal delegate selection.
 *
 * Based on the report recommendations:
 * - Snapdragon: Use QNN Delegate for Hexagon NPU (best performance)
 * - Exynos/Tensor: Use GPU Delegate (NNAPI deprecated in Android 15)
 * - Others: Use GPU Delegate with CPU fallback
 */
object ChipsetDetector {
    private const val TAG = "ChipsetDetector"

    enum class ChipsetType {
        QUALCOMM_SNAPDRAGON,  // Use QNN Delegate
        SAMSUNG_EXYNOS,        // Use GPU Delegate
        GOOGLE_TENSOR,         // Use GPU Delegate
        MEDIATEK,              // Use GPU Delegate
        OTHER                  // Use GPU Delegate with CPU fallback
    }

    /**
     * Detect the chipset type of the current device.
     *
     * Uses android.os.Build.HARDWARE and Build.SOC_MANUFACTURER (API 31+)
     * to determine the chipset vendor.
     */
    fun detectChipset(): ChipsetType {
        val hardware = Build.HARDWARE.lowercase()
        val manufacturer = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            Build.SOC_MANUFACTURER.lowercase()
        } else {
            ""
        }
        val model = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            Build.SOC_MODEL.lowercase()
        } else {
            ""
        }

        Log.d(TAG, "Hardware: $hardware, SoC Manufacturer: $manufacturer, SoC Model: $model")

        return when {
            // Qualcomm Snapdragon detection
            manufacturer.contains("qualcomm") ||
            hardware.contains("qcom") ||
            hardware.contains("snapdragon") ||
            model.contains("snapdragon") -> {
                Log.i(TAG, "Detected Qualcomm Snapdragon - will use QNN Delegate")
                ChipsetType.QUALCOMM_SNAPDRAGON
            }

            // Samsung Exynos detection
            manufacturer.contains("samsung") ||
            hardware.contains("exynos") ||
            hardware.contains("samsungexynos") -> {
                Log.i(TAG, "Detected Samsung Exynos - will use GPU Delegate")
                ChipsetType.SAMSUNG_EXYNOS
            }

            // Google Tensor detection
            manufacturer.contains("google") ||
            hardware.contains("tensor") ||
            model.contains("tensor") -> {
                Log.i(TAG, "Detected Google Tensor - will use GPU Delegate")
                ChipsetType.GOOGLE_TENSOR
            }

            // MediaTek detection
            manufacturer.contains("mediatek") ||
            hardware.contains("mt") ||
            hardware.contains("mediatek") -> {
                Log.i(TAG, "Detected MediaTek - will use GPU Delegate")
                ChipsetType.MEDIATEK
            }

            else -> {
                Log.i(TAG, "Unknown chipset ($hardware) - will use GPU Delegate with CPU fallback")
                ChipsetType.OTHER
            }
        }
    }

    /**
     * Check if the device supports QNN (Qualcomm Neural Network) acceleration.
     */
    fun supportsQNN(): Boolean {
        return detectChipset() == ChipsetType.QUALCOMM_SNAPDRAGON
    }

    /**
     * Get recommended acceleration mode based on chipset.
     */
    fun getRecommendedAcceleration(): String {
        return when (detectChipset()) {
            ChipsetType.QUALCOMM_SNAPDRAGON -> "NPU (QNN Delegate)"
            ChipsetType.SAMSUNG_EXYNOS -> "GPU (OpenCL)"
            ChipsetType.GOOGLE_TENSOR -> "GPU (OpenCL)"
            ChipsetType.MEDIATEK -> "GPU (OpenCL)"
            ChipsetType.OTHER -> "GPU/CPU"
        }
    }

    /**
     * Get device info string for debugging.
     */
    fun getDeviceInfo(): String {
        val sb = StringBuilder()
        sb.appendLine("Device: ${Build.MANUFACTURER} ${Build.MODEL}")
        sb.appendLine("Hardware: ${Build.HARDWARE}")
        sb.appendLine("Board: ${Build.BOARD}")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            sb.appendLine("SoC Manufacturer: ${Build.SOC_MANUFACTURER}")
            sb.appendLine("SoC Model: ${Build.SOC_MODEL}")
        }
        sb.appendLine("Android: ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})")
        sb.appendLine("Chipset Type: ${detectChipset()}")
        sb.appendLine("Recommended: ${getRecommendedAcceleration()}")
        return sb.toString()
    }
}
