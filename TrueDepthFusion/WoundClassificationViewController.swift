import UIKit
import CoreML
import Vision

/// Loads the EdgeNeXt-Small ProtoNet encoder + precomputed class prototypes,
/// runs a captured frame through the encoder, and classifies via nearest-prototype
/// Euclidean distance (matching the training-time ProtoNet inference rule).
final class WoundClassificationViewController: UIViewController {

    // MARK: - Public entry point

    /// Pass in the RGB frame you want classified — e.g. the same frame used
    /// for measurement, or one the user picks from the scan's rgb.mp4.
    var inputImage: UIImage?

    // MARK: - Model state

    private var encoderModel: EdgeNeXtEncoder?          // Xcode-generated class from the .mlpackage
    private var prototypes: [String: [Float]] = [:]     // class name -> embedding vector

    // MARK: - UI

    private let imageView = UIImageView()
    private let resultLabel = UILabel()
    private let distanceLabel = UILabel()
    private let classifyButton = UIButton(type: .system)
    private let activityIndicator = UIActivityIndicatorView(style: .large)

    override func viewDidLoad() {
        super.viewDidLoad()
        title = "Classify Wound"
        view.backgroundColor = .systemBackground

        setupUI()
        loadModelAndPrototypes()

        if let img = inputImage {
            imageView.image = img
        }
    }

    // MARK: - Setup

    private func setupUI() {
        imageView.contentMode = .scaleAspectFit
        imageView.backgroundColor = .secondarySystemBackground
        imageView.layer.cornerRadius = 8
        imageView.clipsToBounds = true

        resultLabel.font = .systemFont(ofSize: 20, weight: .semibold)
        resultLabel.textAlignment = .center
        resultLabel.numberOfLines = 0
        resultLabel.text = "No prediction yet"

        distanceLabel.font = .systemFont(ofSize: 14, weight: .regular)
        distanceLabel.textColor = .secondaryLabel
        distanceLabel.textAlignment = .center
        distanceLabel.numberOfLines = 0

        classifyButton.setTitle("Classify Wound", for: .normal)
        classifyButton.titleLabel?.font = .systemFont(ofSize: 17, weight: .medium)
        classifyButton.addTarget(self, action: #selector(classifyTapped), for: .touchUpInside)

        activityIndicator.hidesWhenStopped = true

        let stack = UIStackView(arrangedSubviews: [
            imageView, classifyButton, activityIndicator, resultLabel, distanceLabel
        ])
        stack.axis = .vertical
        stack.spacing = 16
        stack.alignment = .fill
        stack.translatesAutoresizingMaskIntoConstraints = false

        view.addSubview(stack)

        NSLayoutConstraint.activate([
            stack.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 24),
            stack.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            stack.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            imageView.heightAnchor.constraint(equalToConstant: 280),
        ])
    }

    // MARK: - Load model + prototypes

    private func loadModelAndPrototypes() {
        // Load the Core ML encoder
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all // let CoreML pick ANE/GPU/CPU as available
            encoderModel = try EdgeNeXtEncoder(configuration: config)
        } catch {
            resultLabel.text = "Failed to load model: \(error.localizedDescription)"
            classifyButton.isEnabled = false
            return
        }

        // Load prototypes.json bundled as an app resource
        guard let url = Bundle.main.url(forResource: "prototypes", withExtension: "json") else {
            resultLabel.text = "prototypes.json not found in bundle"
            classifyButton.isEnabled = false
            return
        }

        do {
            let data = try Data(contentsOf: url)
            let decoded = try JSONDecoder().decode([String: [Float]].self, from: data)
            prototypes = decoded
        } catch {
            resultLabel.text = "Failed to parse prototypes.json: \(error.localizedDescription)"
            classifyButton.isEnabled = false
        }
    }

    // MARK: - Classification

    @objc private func classifyTapped() {
        guard let image = inputImage ?? imageView.image else {
            resultLabel.text = "No image to classify"
            return
        }
        guard let encoderModel else {
            resultLabel.text = "Model not loaded"
            return
        }
        guard !prototypes.isEmpty else {
            resultLabel.text = "Prototypes not loaded"
            return
        }

        activityIndicator.startAnimating()
        classifyButton.isEnabled = false

        // Run inference off the main thread — Core ML calls can take noticeable time on first invocation
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }

            guard let pixelBuffer = image.toCVPixelBuffer(width: 256, height: 256) else {
                DispatchQueue.main.async {
                    self.activityIndicator.stopAnimating()
                    self.classifyButton.isEnabled = true
                    self.resultLabel.text = "Failed to prepare image"
                }
                return
            }

            do {
                let output = try encoderModel.prediction(input_image: pixelBuffer)
                let embedding = output.embedding // MLMultiArray

                let queryVector = embedding.toFloatArray()
                let (predictedClass, distance, allDistances) = self.nearestPrototype(for: queryVector)

                DispatchQueue.main.async {
                    self.activityIndicator.stopAnimating()
                    self.classifyButton.isEnabled = true
                    self.resultLabel.text = predictedClass
                    self.distanceLabel.text = self.formatDistances(allDistances, top: 3)
                }
            } catch {
                DispatchQueue.main.async {
                    self.activityIndicator.stopAnimating()
                    self.classifyButton.isEnabled = true
                    self.resultLabel.text = "Inference failed: \(error.localizedDescription)"
                }
            }
        }
    }

    /// Nearest-prototype lookup using Euclidean distance, matching the
    /// checkpoint's `distance: "euclidean"` training configuration.
    private func nearestPrototype(for query: [Float]) -> (className: String, distance: Float, all: [(String, Float)]) {
        var distances: [(String, Float)] = []

        for (className, protoVector) in prototypes {
            guard protoVector.count == query.count else {
                continue // dimension mismatch, skip defensively
            }
            var sumSquares: Float = 0
            for i in 0..<query.count {
                let diff = query[i] - protoVector[i]
                sumSquares += diff * diff
            }
            let dist = sqrt(sumSquares)
            distances.append((className, dist))
        }

        distances.sort { $0.1 < $1.1 }

        guard let best = distances.first else {
            return ("Unknown", .infinity, [])
        }
        return (best.0, best.1, distances)
    }

    private func formatDistances(_ distances: [(String, Float)], top: Int) -> String {
        let topN = distances.prefix(top)
        return topN.map { String(format: "%@: %.2f", $0.0, $0.1) }.joined(separator: "\n")
    }
}

// MARK: - Helpers

private extension UIImage {
    /// Resizes and converts to a CVPixelBuffer matching the Core ML model's expected input.
    /// Note: normalization (mean/std subtraction) is baked into the .mlpackage itself via
    /// the ImageType scale/bias set during coremltools conversion, so this only needs to
    /// handle resize + pixel format, not manual normalization.
    func toCVPixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        ]
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height,
            kCVPixelFormatType_32ARGB, attrs as CFDictionary, &pixelBuffer
        )
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width, height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ), let cgImage = self.cgImage else { return nil }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buffer
    }
}

private extension MLMultiArray {
    /// Flattens an MLMultiArray embedding output into a plain [Float] for distance math.
    func toFloatArray() -> [Float] {
        let count = self.count
        var result = [Float](repeating: 0, count: count)
        let ptr = self.dataPointer.bindMemory(to: Float32.self, capacity: count)
        for i in 0..<count {
            result[i] = ptr[i]
        }
        return result
    }
}
