import Foundation
import Vision
import UIKit
import CoreMedia
import CoreVideo

/// Detector error types.
enum DetectorError: Error {
    case notInitialized
    case invalidImageFormat
    case fileNotFound
    case inferenceFailed(String)
}

/// Vision Framework-based pose detector.
///
/// Uses VNDetectHumanBodyPoseRequest for pose detection with automatic
/// Neural Engine acceleration on supported devices.
class VisionPoseDetector {

    // MARK: - Properties

    private var config: DetectorConfig
    private let landmarkMapper = LandmarkMapper()
    private var isReady = false

    // MARK: - Initialization

    init(config: DetectorConfig) {
        self.config = config
    }

    /// Initialize the detector.
    ///
    /// On iOS, Vision Framework is always available and automatically
    /// uses Neural Engine when available.
    func initialize() throws -> AccelerationMode {
        isReady = true
        // Vision Framework automatically uses Neural Engine on A12+ chips
        return .npu
    }

    /// Update the detector configuration.
    func updateConfig(_ config: DetectorConfig) {
        self.config = config
    }

    /// Release resources.
    func dispose() {
        isReady = false
    }

    // MARK: - Detection from Data

    /// Detect poses in image data.
    ///
    /// - Parameter imageData: JPEG or PNG encoded image data
    /// - Returns: PoseResult with detected poses
    func detectPose(imageData: Data) throws -> PoseResult {
        guard isReady else {
            throw DetectorError.notInitialized
        }

        guard let uiImage = UIImage(data: imageData),
              let cgImage = uiImage.cgImage else {
            throw DetectorError.invalidImageFormat
        }

        return try detectPose(cgImage: cgImage)
    }

    // MARK: - Detection from File

    /// Detect poses from an image file.
    ///
    /// - Parameter filePath: Path to the image file
    /// - Returns: PoseResult with detected poses
    func detectPose(filePath: String) throws -> PoseResult {
        guard isReady else {
            throw DetectorError.notInitialized
        }

        guard FileManager.default.fileExists(atPath: filePath) else {
            throw DetectorError.fileNotFound
        }

        guard let uiImage = UIImage(contentsOfFile: filePath),
              let cgImage = uiImage.cgImage else {
            throw DetectorError.invalidImageFormat
        }

        return try detectPose(cgImage: cgImage)
    }

    // MARK: - Frame Processing for Camera Stream

    /// Process a camera sample buffer for realtime detection.
    ///
    /// - Parameter sampleBuffer: CMSampleBuffer from camera
    /// - Returns: PoseResult with detected poses
    func processFrame(_ sampleBuffer: CMSampleBuffer) throws -> PoseResult {
        guard isReady else {
            throw DetectorError.notInitialized
        }

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            throw DetectorError.invalidImageFormat
        }

        return try processFrame(pixelBuffer: pixelBuffer)
    }

    /// Process a CVPixelBuffer for realtime detection.
    ///
    /// - Parameter pixelBuffer: CVPixelBuffer from camera
    /// - Returns: PoseResult with detected poses
    func processFrame(pixelBuffer: CVPixelBuffer) throws -> PoseResult {
        guard isReady else {
            throw DetectorError.notInitialized
        }

        let startTime = CFAbsoluteTimeGetCurrent()

        let request = VNDetectHumanBodyPoseRequest()

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])

        do {
            try handler.perform([request])
        } catch {
            throw DetectorError.inferenceFailed(error.localizedDescription)
        }

        let processingTime = Int((CFAbsoluteTimeGetCurrent() - startTime) * 1000)

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        guard let observations = request.results else {
            return PoseResult.empty(
                processingTimeMs: processingTime,
                imageWidth: width,
                imageHeight: height
            )
        }

        // Filter and map observations
        let poses = observations
            .prefix(config.maxPoses)
            .filter { $0.confidence >= config.minConfidence }
            .map { observation in
                landmarkMapper.mapVisionToPose(observation)
            }

        return PoseResult(
            poses: poses,
            processingTimeMs: processingTime,
            accelerationMode: .npu,
            timestamp: Date(),
            imageWidth: width,
            imageHeight: height
        )
    }

    // MARK: - Core Detection

    /// Perform pose detection on a CGImage.
    private func detectPose(cgImage: CGImage) throws -> PoseResult {
        let startTime = CFAbsoluteTimeGetCurrent()

        let request = VNDetectHumanBodyPoseRequest()

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])

        do {
            try handler.perform([request])
        } catch {
            throw DetectorError.inferenceFailed(error.localizedDescription)
        }

        let processingTime = Int((CFAbsoluteTimeGetCurrent() - startTime) * 1000)

        guard let observations = request.results else {
            return PoseResult.empty(
                processingTimeMs: processingTime,
                imageWidth: cgImage.width,
                imageHeight: cgImage.height
            )
        }

        // Filter and map observations
        let poses = observations
            .prefix(config.maxPoses)
            .filter { $0.confidence >= config.minConfidence }
            .map { observation in
                landmarkMapper.mapVisionToPose(observation)
            }

        return PoseResult(
            poses: poses,
            processingTimeMs: processingTime,
            accelerationMode: .npu,
            timestamp: Date(),
            imageWidth: cgImage.width,
            imageHeight: cgImage.height
        )
    }
}
