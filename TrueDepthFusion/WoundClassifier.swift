//
//  WoundClassifier.swift
//  TrueDepthFusion
//
//  Loads the EdgeNeXt-Small ProtoNet encoder + precomputed class prototypes,
//  and classifies a UIImage via nearest-prototype Euclidean distance —
//  matching the training-time ProtoNet inference rule.
//
//  Requires EdgeNeXtEncoder.mlpackage and prototypes.json to be added to this
//  target (same files used in the woundcarescanner integration).
//

import UIKit
import CoreML

final class WoundClassifier {

    struct Prediction {
        let label: String
        let distance: Float
    }

    enum ClassifierError: Error, LocalizedError {
        case modelLoadFailed(String)
        case prototypesLoadFailed(String)
        case imagePrepFailed
        case inferenceFailed(String)

        var errorDescription: String? {
            switch self {
            case .modelLoadFailed(let msg): return "Failed to load model: \(msg)"
            case .prototypesLoadFailed(let msg): return "Failed to load prototypes: \(msg)"
            case .imagePrepFailed: return "Failed to prepare image for the model"
            case .inferenceFailed(let msg): return "Inference failed: \(msg)"
            }
        }
    }

    private let encoderModel: EdgeNeXtEncoder
    private let prototypes: [String: [Float]]

    init() throws {
        let config = MLModelConfiguration()
        #if targetEnvironment(simulator)
        // The Simulator's Espresso/MPSGraph backend can throw on this model;
        // CPU-only inference is reliable there. Real devices use .all (ANE/GPU/CPU).
        config.computeUnits = .cpuOnly
        #else
        config.computeUnits = .all
        #endif

        do {
            encoderModel = try EdgeNeXtEncoder(configuration: config)
        } catch {
            throw ClassifierError.modelLoadFailed(error.localizedDescription)
        }

        guard let url = Bundle.main.url(forResource: "prototypes", withExtension: "json") else {
            throw ClassifierError.prototypesLoadFailed("prototypes.json not found in app bundle")
        }

        do {
            let data = try Data(contentsOf: url)
            prototypes = try JSONDecoder().decode([String: [Float]].self, from: data)
        } catch {
            throw ClassifierError.prototypesLoadFailed(error.localizedDescription)
        }
    }

    /// Runs the encoder on `image` and returns every class ranked by ascending
    /// distance (closest / most-likely prototype first).
    func classify(_ image: UIImage) throws -> [Prediction] {
        guard let cgImage = image.cgImage else {
            throw ClassifierError.imagePrepFailed
        }

        // Let Core ML handle resize/crop/format conversion itself, using the
        // model's own declared input constraint — guarantees byte-for-byte the
        // same preprocessing Core ML expects (avoids hand-rolled buffer bugs).
        guard let constraint = encoderModel.model.modelDescription
            .inputDescriptionsByName["input_image"]?.imageConstraint else {
            throw ClassifierError.imagePrepFailed
        }

        let featureValue: MLFeatureValue
        do {
            featureValue = try MLFeatureValue(cgImage: cgImage, constraint: constraint, options: nil)
        } catch {
            throw ClassifierError.imagePrepFailed
        }

        guard let pixelBuffer = featureValue.imageBufferValue else {
            throw ClassifierError.imagePrepFailed
        }

        let output: EdgeNeXtEncoderOutput
        do {
            output = try encoderModel.prediction(input_image: pixelBuffer)
        } catch {
            throw ClassifierError.inferenceFailed(error.localizedDescription)
        }

        let embedding = output.embedding // MLMultiArray, shape [1, 304]
        let embeddingValues = (0..<embedding.count).map { embedding[$0].floatValue }

        let ranked = prototypes.map { label, prototype -> Prediction in
            Prediction(label: label, distance: _euclideanDistance(embeddingValues, prototype))
        }.sorted { $0.distance < $1.distance }

        return ranked
    }

    private func _euclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return .greatestFiniteMagnitude }
        var sum: Float = 0
        for i in 0..<a.count {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        return sqrt(sum)
    }
}
