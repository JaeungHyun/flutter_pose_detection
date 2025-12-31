import Foundation
import UIKit
import CoreMedia
import CoreVideo
import TensorFlowLite

/// Detector error types.
enum DetectorError: Error {
    case notInitialized
    case invalidImageFormat
    case fileNotFound
    case inferenceFailed(String)
}

/// LiteRT-based pose detector using MoveNet with CoreML/Metal/CPU acceleration.
///
/// Initialization order:
/// 1. Native Core ML HRNetPose (best performance on A12+)
/// 2. TFLite HRNetPose with CoreML delegate
/// 3. TFLite HRNetPose with Metal delegate
/// 4. TFLite MoveNet fallback
class LiteRtPoseDetector {

    // MARK: - Constants

    private static let TAG = "LiteRtPoseDetector"

    // HRNetPose - optimized for Neural Engine (256x192 input, heatmap output)
    private static let MODEL_HRNETPOSE = "hrnetpose_w8a8"
    private static let HRNET_INPUT_WIDTH = 192
    private static let HRNET_INPUT_HEIGHT = 256
    private static let HRNET_HEATMAP_WIDTH = 48
    private static let HRNET_HEATMAP_HEIGHT = 64

    // MoveNet - fallback models (square input, direct keypoint output)
    private static let MODEL_LIGHTNING = "movenet_lightning"
    private static let MODEL_THUNDER = "movenet_thunder"
    private static let INPUT_SIZE_LIGHTNING = 192
    private static let INPUT_SIZE_THUNDER = 256

    private static let NUM_KEYPOINTS = 17

    // MARK: - Properties

    private var interpreter: Interpreter?
    private var coreMLDetector: CoreMLPoseDetector?  // Native CoreML detector
    private var config: DetectorConfig
    private var _accelerationMode: AccelerationMode = .unknown
    private var isReady = false
    private var useHRNetPose = false
    private var useNativeCoreML = false  // True when using native CoreML

    var accelerationMode: AccelerationMode { _accelerationMode }
    var isInitialized: Bool { isReady && (interpreter != nil || coreMLDetector != nil) }

    // MARK: - Initialization

    init(config: DetectorConfig) {
        self.config = config
    }

    /// Initialize the detector with hardware acceleration.
    ///
    /// Initialization order:
    /// 1. Native Core ML HRNetPose (best performance on A12+)
    /// 2. TFLite HRNetPose with CoreML delegate
    /// 3. TFLite HRNetPose with Metal delegate
    /// 4. TFLite MoveNet fallback
    ///
    /// - Returns: The acceleration mode being used
    func initialize() throws -> AccelerationMode {
        // Try native Core ML HRNetPose first (best ANE performance)
        if config.preferredAcceleration != .cpu {
            print("[\(Self.TAG)] Trying native Core ML HRNetPose...")
            coreMLDetector = CoreMLPoseDetector(config: config)
            if let mode = try? coreMLDetector?.initialize() {
                useNativeCoreML = true
                useHRNetPose = true
                _accelerationMode = mode
                isReady = true
                print("[\(Self.TAG)] ✓ Initialized native Core ML HRNetPose (ANE optimized)")
                return _accelerationMode
            } else {
                print("[\(Self.TAG)] Native Core ML not available, trying TFLite...")
                coreMLDetector = nil
            }
        }

        // Try TFLite HRNetPose with CoreML/Metal
        if config.preferredAcceleration != .cpu {
            print("[\(Self.TAG)] Looking for TFLite HRNetPose model: \(Self.MODEL_HRNETPOSE)")
            if let hrnetPath = getModelPath(Self.MODEL_HRNETPOSE) {
                print("[\(Self.TAG)] Found HRNetPose at: \(hrnetPath)")
                // Try CoreML (Neural Engine)
                if config.preferredAcceleration != .gpu {
                    if let coreMLInterpreter = try? createInterpreterWithCoreML(modelPath: hrnetPath) {
                        interpreter = coreMLInterpreter
                        useHRNetPose = true
                        _accelerationMode = .npu
                        isReady = true
                        print("[\(Self.TAG)] Initialized HRNetPose with TFLite CoreML delegate")
                        return _accelerationMode
                    }
                    print("[\(Self.TAG)] HRNetPose CoreML failed, trying Metal")
                }

                // Try Metal (GPU)
                if let metalInterpreter = try? createInterpreterWithMetal(modelPath: hrnetPath) {
                    interpreter = metalInterpreter
                    useHRNetPose = true
                    _accelerationMode = .gpu
                    isReady = true
                    print("[\(Self.TAG)] Initialized HRNetPose with TFLite Metal delegate")
                    return _accelerationMode
                }
                print("[\(Self.TAG)] HRNetPose Metal failed, falling back to MoveNet")
            } else {
                print("[\(Self.TAG)] HRNetPose model not found, falling back to MoveNet")
            }
        }

        // Fallback to MoveNet
        let modelFileName = getModelFileName()
        guard let modelPath = getModelPath(modelFileName) else {
            throw DetectorError.inferenceFailed("Model file not found: \(modelFileName)")
        }

        // Try CoreML delegate first (Neural Engine on A12+)
        if config.preferredAcceleration != .cpu && config.preferredAcceleration != .gpu {
            if let coreMLInterpreter = try? createInterpreterWithCoreML(modelPath: modelPath) {
                interpreter = coreMLInterpreter
                useHRNetPose = false
                _accelerationMode = .npu
                isReady = true
                print("[\(Self.TAG)] Initialized MoveNet with CoreML (Neural Engine)")
                return _accelerationMode
            }
            print("[\(Self.TAG)] CoreML delegate failed, trying Metal")
        }

        // Try Metal delegate (GPU)
        if config.preferredAcceleration != .cpu {
            if let metalInterpreter = try? createInterpreterWithMetal(modelPath: modelPath) {
                interpreter = metalInterpreter
                useHRNetPose = false
                _accelerationMode = .gpu
                isReady = true
                print("[\(Self.TAG)] Initialized MoveNet with Metal (GPU)")
                return _accelerationMode
            }
            print("[\(Self.TAG)] Metal delegate failed, falling back to CPU")
        }

        // Fall back to CPU
        interpreter = try createInterpreterCPU(modelPath: modelPath)
        useHRNetPose = false
        _accelerationMode = .cpu
        isReady = true
        print("[\(Self.TAG)] Initialized MoveNet with CPU")

        return _accelerationMode
    }

    // MARK: - Interpreter Creation

    private func createInterpreterWithCoreML(modelPath: String) throws -> Interpreter {
        var options = Interpreter.Options()
        options.threadCount = 4

        guard let coreMLDelegate = CoreMLDelegate() else {
            throw DetectorError.inferenceFailed("Failed to create CoreML delegate")
        }
        let interpreter = try Interpreter(modelPath: modelPath, options: options, delegates: [coreMLDelegate])
        try interpreter.allocateTensors()
        return interpreter
    }

    private func createInterpreterWithMetal(modelPath: String) throws -> Interpreter {
        var options = Interpreter.Options()
        options.threadCount = 4

        let metalDelegate = MetalDelegate()
        let interpreter = try Interpreter(modelPath: modelPath, options: options, delegates: [metalDelegate])
        try interpreter.allocateTensors()
        return interpreter
    }

    private func createInterpreterCPU(modelPath: String) throws -> Interpreter {
        var options = Interpreter.Options()
        options.threadCount = 4

        let interpreter = try Interpreter(modelPath: modelPath, options: options)
        try interpreter.allocateTensors()
        return interpreter
    }

    // MARK: - Model Path

    private func getModelFileName() -> String {
        switch config.mode {
        case .fast, .balanced:
            return Self.MODEL_LIGHTNING
        case .accurate:
            return Self.MODEL_THUNDER
        }
    }

    private func getModelPath(_ modelName: String) -> String? {
        let podBundle = Bundle(for: type(of: self))

        // Try resource bundle first (CocoaPods resource_bundles)
        if let resourceBundlePath = podBundle.path(forResource: "flutter_pose_detection", ofType: "bundle"),
           let resourceBundle = Bundle(path: resourceBundlePath),
           let modelPath = resourceBundle.path(forResource: modelName, ofType: "tflite") {
            print("[\(Self.TAG)] Found model in resource bundle: \(modelPath)")
            return modelPath
        }

        // Try plugin bundle directly (Pod resources)
        if let bundlePath = podBundle.path(forResource: modelName, ofType: "tflite") {
            print("[\(Self.TAG)] Found model in pod bundle: \(bundlePath)")
            return bundlePath
        }

        // Try main bundle
        if let mainPath = Bundle.main.path(forResource: modelName, ofType: "tflite") {
            print("[\(Self.TAG)] Found model in main bundle: \(mainPath)")
            return mainPath
        }

        print("[\(Self.TAG)] Model not found: \(modelName).tflite")
        return nil
    }

    // MARK: - Configuration

    func updateConfig(_ config: DetectorConfig) {
        self.config = config
    }

    func dispose() {
        interpreter = nil
        coreMLDetector?.dispose()
        coreMLDetector = nil
        isReady = false
        useHRNetPose = false
        useNativeCoreML = false
        _accelerationMode = .unknown
        print("[\(Self.TAG)] Detector disposed")
    }

    // MARK: - Detection

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

    // MARK: - Frame Processing

    /// Process a CVPixelBuffer for realtime detection.
    ///
    /// - Parameter pixelBuffer: CVPixelBuffer from camera
    /// - Returns: PoseResult with detected poses
    func processFrame(pixelBuffer: CVPixelBuffer) throws -> PoseResult {
        guard isReady else {
            throw DetectorError.notInitialized
        }

        // Use native CoreML if available (best performance)
        if useNativeCoreML, let detector = coreMLDetector {
            return try detector.processFrame(pixelBuffer: pixelBuffer)
        }

        // Fall back to TFLite
        let startTime = CFAbsoluteTimeGetCurrent()
        let (inputWidth, inputHeight) = getInputDimensions()

        // Convert pixel buffer to RGB data
        guard let rgbData = pixelBufferToRGBData(pixelBuffer, width: inputWidth, height: inputHeight) else {
            throw DetectorError.invalidImageFormat
        }

        // Run inference
        let keypoints = try runInference(rgbData: rgbData, width: inputWidth, height: inputHeight)

        let processingTime = Int((CFAbsoluteTimeGetCurrent() - startTime) * 1000)

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        // Map to landmarks
        let landmarks = mapKeypoints(keypoints)

        // Calculate score
        let detectedLandmarks = landmarks.filter { $0.isDetected }
        let score = detectedLandmarks.isEmpty ? 0.0 : detectedLandmarks.map { $0.visibility }.reduce(0, +) / Double(detectedLandmarks.count)

        let pose: Pose?
        if score >= Double(config.minConfidence) {
            pose = Pose(
                landmarks: landmarks,
                score: score,
                boundingBox: calculateBoundingBox(landmarks)
            )
        } else {
            pose = nil
        }

        return PoseResult(
            poses: pose.map { [$0] } ?? [],
            processingTimeMs: processingTime,
            accelerationMode: _accelerationMode,
            timestamp: Date(),
            imageWidth: width,
            imageHeight: height
        )
    }

    // MARK: - Core Detection

    private func detectPose(cgImage: CGImage) throws -> PoseResult {
        // Use native CoreML if available (best performance)
        if useNativeCoreML, let detector = coreMLDetector {
            return try detector.detectPose(cgImage: cgImage)
        }

        // Fall back to TFLite
        let startTime = CFAbsoluteTimeGetCurrent()
        let (inputWidth, inputHeight) = getInputDimensions()

        // Convert to RGB data
        guard let rgbData = cgImageToRGBData(cgImage, width: inputWidth, height: inputHeight) else {
            throw DetectorError.invalidImageFormat
        }

        // Run inference
        let keypoints = try runInference(rgbData: rgbData, width: inputWidth, height: inputHeight)

        let processingTime = Int((CFAbsoluteTimeGetCurrent() - startTime) * 1000)

        // Map to landmarks
        let landmarks = mapKeypoints(keypoints)

        // Calculate score
        let detectedLandmarks = landmarks.filter { $0.isDetected }
        let score = detectedLandmarks.isEmpty ? 0.0 : detectedLandmarks.map { $0.visibility }.reduce(0, +) / Double(detectedLandmarks.count)

        let pose: Pose?
        if score >= Double(config.minConfidence) {
            pose = Pose(
                landmarks: landmarks,
                score: score,
                boundingBox: calculateBoundingBox(landmarks)
            )
        } else {
            pose = nil
        }

        return PoseResult(
            poses: pose.map { [$0] } ?? [],
            processingTimeMs: processingTime,
            accelerationMode: _accelerationMode,
            timestamp: Date(),
            imageWidth: cgImage.width,
            imageHeight: cgImage.height
        )
    }

    // MARK: - Inference

    private func runInference(rgbData: Data, width: Int, height: Int) throws -> [[Float]] {
        guard let interpreter = interpreter else {
            throw DetectorError.notInitialized
        }

        // Copy input data
        try interpreter.copy(rgbData, toInputAt: 0)

        // Run inference
        try interpreter.invoke()

        // Get output tensor
        let outputTensor = try interpreter.output(at: 0)
        let outputData = outputTensor.data

        if useHRNetPose {
            return parseHRNetPoseOutput(outputData)
        } else {
            return parseMoveNetOutput(outputData)
        }
    }

    /// Parse HRNetPose heatmap output: [1, 17, 64, 48] -> keypoints
    private func parseHRNetPoseOutput(_ outputData: Data) -> [[Float]] {
        let floats = outputData.withUnsafeBytes { pointer in
            Array(pointer.bindMemory(to: Float.self))
        }

        let heatmapSize = Self.HRNET_HEATMAP_WIDTH * Self.HRNET_HEATMAP_HEIGHT
        var keypoints = [[Float]]()

        for k in 0..<Self.NUM_KEYPOINTS {
            var maxVal: Float = -.greatestFiniteMagnitude
            var maxIdx = 0

            for i in 0..<heatmapSize {
                let idx = k * heatmapSize + i
                if idx < floats.count && floats[idx] > maxVal {
                    maxVal = floats[idx]
                    maxIdx = i
                }
            }

            let heatmapX = maxIdx % Self.HRNET_HEATMAP_WIDTH
            let heatmapY = maxIdx / Self.HRNET_HEATMAP_WIDTH
            let normalizedX = Float(heatmapX) / Float(Self.HRNET_HEATMAP_WIDTH)
            let normalizedY = Float(heatmapY) / Float(Self.HRNET_HEATMAP_HEIGHT)
            let confidence = 1.0 / (1.0 + exp(-maxVal))  // sigmoid

            keypoints.append([normalizedY, normalizedX, confidence])
        }

        return keypoints
    }

    /// Parse MoveNet output: [1, 1, 17, 3] -> keypoints
    private func parseMoveNetOutput(_ outputData: Data) -> [[Float]] {
        let floats = outputData.withUnsafeBytes { pointer in
            Array(pointer.bindMemory(to: Float.self))
        }

        var keypoints = [[Float]]()
        for i in 0..<Self.NUM_KEYPOINTS {
            let y = floats[i * 3]
            let x = floats[i * 3 + 1]
            let confidence = floats[i * 3 + 2]
            keypoints.append([y, x, confidence])
        }

        return keypoints
    }

    // MARK: - Image Processing

    private func cgImageToRGBData(_ image: CGImage, width: Int, height: Int) -> Data? {
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8

        var pixelData = [UInt8](repeating: 0, count: height * bytesPerRow)
        let colorSpace = CGColorSpaceCreateDeviceRGB()

        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else {
            return nil
        }

        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        // HRNetPose uses UINT8 input, MoveNet uses normalized float
        if useHRNetPose {
            // UINT8 format for HRNetPose (quantized model)
            var uint8Data = [UInt8](repeating: 0, count: width * height * 3)
            for y in 0..<height {
                for x in 0..<width {
                    let offset = y * bytesPerRow + x * bytesPerPixel
                    let uint8Offset = (y * width + x) * 3
                    uint8Data[uint8Offset] = pixelData[offset]      // R
                    uint8Data[uint8Offset + 1] = pixelData[offset + 1] // G
                    uint8Data[uint8Offset + 2] = pixelData[offset + 2] // B
                }
            }
            return Data(uint8Data)
        } else {
            // Float format normalized to [-1, 1] for MoveNet
            var floatData = [Float](repeating: 0, count: width * height * 3)
            for y in 0..<height {
                for x in 0..<width {
                    let offset = y * bytesPerRow + x * bytesPerPixel
                    let r = Float(pixelData[offset]) / 127.5 - 1.0
                    let g = Float(pixelData[offset + 1]) / 127.5 - 1.0
                    let b = Float(pixelData[offset + 2]) / 127.5 - 1.0

                    let floatOffset = (y * width + x) * 3
                    floatData[floatOffset] = r
                    floatData[floatOffset + 1] = g
                    floatData[floatOffset + 2] = b
                }
            }
            return Data(bytes: floatData, count: floatData.count * MemoryLayout<Float>.size)
        }
    }

    private func pixelBufferToRGBData(_ pixelBuffer: CVPixelBuffer, width targetWidth: Int, height targetHeight: Int) -> Data? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return nil
        }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)

        // Create a CGImage from the pixel buffer
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var bitmapInfo: UInt32

        switch pixelFormat {
        case kCVPixelFormatType_32BGRA:
            bitmapInfo = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
        case kCVPixelFormatType_32ARGB:
            bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
        default:
            bitmapInfo = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
        }

        guard let context = CGContext(
            data: baseAddress,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ),
        let cgImage = context.makeImage() else {
            return nil
        }

        return cgImageToRGBData(cgImage, width: targetWidth, height: targetHeight)
    }

    // MARK: - Keypoint Mapping

    /// MoveNet to MediaPipe index mapping
    private let moveNetToMediaPipe: [Int: Int] = [
        0: 0,   // nose
        1: 2,   // left_eye
        2: 5,   // right_eye
        3: 7,   // left_ear
        4: 8,   // right_ear
        5: 11,  // left_shoulder
        6: 12,  // right_shoulder
        7: 13,  // left_elbow
        8: 14,  // right_elbow
        9: 15,  // left_wrist
        10: 16, // right_wrist
        11: 23, // left_hip
        12: 24, // right_hip
        13: 25, // left_knee
        14: 26, // right_knee
        15: 27, // left_ankle
        16: 28  // right_ankle
    ]

    private func mapKeypoints(_ keypoints: [[Float]]) -> [PoseLandmark] {
        var landmarks = (0..<LandmarkType.count).map { index in
            PoseLandmark.notDetected(type: LandmarkType(rawValue: index)!)
        }

        for (moveNetIndex, point) in keypoints.enumerated() {
            guard let mediaPipeIndex = moveNetToMediaPipe[moveNetIndex] else { continue }

            let y = Double(point[0])
            let x = Double(point[1])
            let confidence = Double(point[2])

            if confidence >= Double(config.minConfidence) {
                landmarks[mediaPipeIndex] = PoseLandmark(
                    type: LandmarkType(rawValue: mediaPipeIndex)!,
                    x: x,
                    y: y,
                    z: 0.0,
                    visibility: confidence
                )
            }
        }

        // Interpolate heels
        interpolateHeels(&landmarks)

        return landmarks
    }

    private func interpolateHeels(_ landmarks: inout [PoseLandmark]) {
        // Left heel (index 29) from left knee (25) and left ankle (27)
        if landmarks[25].isDetected && landmarks[27].isDetected {
            landmarks[29] = interpolatePoint(from: landmarks[25], through: landmarks[27], index: 29)
        }

        // Right heel (index 30) from right knee (26) and right ankle (28)
        if landmarks[26].isDetected && landmarks[28].isDetected {
            landmarks[30] = interpolatePoint(from: landmarks[26], through: landmarks[28], index: 30)
        }
    }

    private func interpolatePoint(from: PoseLandmark, through: PoseLandmark, index: Int) -> PoseLandmark {
        let dx = through.x - from.x
        let dy = through.y - from.y
        let extensionFactor = 0.15

        return PoseLandmark(
            type: LandmarkType(rawValue: index)!,
            x: through.x + dx * extensionFactor,
            y: through.y + dy * extensionFactor,
            z: through.z,
            visibility: min(from.visibility, through.visibility) * 0.8
        )
    }

    // MARK: - Utilities

    /// Returns (width, height) for the current model
    private func getInputDimensions() -> (width: Int, height: Int) {
        if useHRNetPose {
            return (Self.HRNET_INPUT_WIDTH, Self.HRNET_INPUT_HEIGHT)
        }
        let size = getMoveNetInputSize()
        return (size, size)
    }

    private func getMoveNetInputSize() -> Int {
        switch config.mode {
        case .fast, .balanced:
            return Self.INPUT_SIZE_LIGHTNING
        case .accurate:
            return Self.INPUT_SIZE_THUNDER
        }
    }

    private func calculateBoundingBox(_ landmarks: [PoseLandmark]) -> BoundingBox? {
        let detected = landmarks.filter { $0.isDetected }
        guard !detected.isEmpty else { return nil }

        let minX = detected.map { $0.x }.min()!
        let maxX = detected.map { $0.x }.max()!
        let minY = detected.map { $0.y }.min()!
        let maxY = detected.map { $0.y }.max()!

        let padding = 0.1
        let width = maxX - minX
        let height = maxY - minY

        return BoundingBox(
            left: max(0.0, min(1.0, minX - width * padding)),
            top: max(0.0, min(1.0, minY - height * padding)),
            width: max(0.0, min(1.0, width * (1 + 2 * padding))),
            height: max(0.0, min(1.0, height * (1 + 2 * padding)))
        )
    }
}
