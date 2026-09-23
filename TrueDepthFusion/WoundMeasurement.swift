//
//  WoundMeasurement.swift
//  TrueDepthFusion
//
//  Model + persistence for beta wound measurements. Measurements are saved as a
//  JSON sidecar next to each scan's .ply file, following the same pattern as the
//  scan's .jpeg thumbnail (same base filename, different extension).
//

import Foundation
import StandardCyborgFusion
import simd

struct WoundMeasurement: Codable {
    let pointA: [Float]  // [x, y, z], in the point cloud's local coordinate space
    let pointB: [Float]
    let distanceMeters: Float
    let dateCreated: Date

    init(pointA: SIMD3<Float>, pointB: SIMD3<Float>, dateCreated: Date = Date()) {
        self.pointA = [pointA.x, pointA.y, pointA.z]
        self.pointB = [pointB.x, pointB.y, pointB.z]
        self.distanceMeters = simd_distance(pointA, pointB)
        self.dateCreated = dateCreated
    }

    var pointAVector: SIMD3<Float> { SIMD3<Float>(pointA[0], pointA[1], pointA[2]) }
    var pointBVector: SIMD3<Float> { SIMD3<Float>(pointB[0], pointB[1], pointB[2]) }
    var distanceCentimeters: Float { distanceMeters * 100 }
    var distanceInches: Float { distanceMeters * 39.3701 }
}

extension Scan {
    /// Sidecar JSON path for this scan's saved measurements. Mirrors the thumbnail's
    /// naming pattern. Returns nil until the scan itself has a plyPath (i.e. has been
    /// written to disk via AppDelegate.add(_:)) — there's nowhere to save alongside yet.
    var measurementsPath: String? {
        guard let plyPath = plyPath else { return nil }
        return (plyPath as NSString).deletingPathExtension + "-measurements.json"
    }

    func loadMeasurements() -> [WoundMeasurement] {
        guard let path = measurementsPath,
              let data = FileManager.default.contents(atPath: path) else {
            return []
        }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return (try? decoder.decode([WoundMeasurement].self, from: data)) ?? []
    }

    @discardableResult
    func saveMeasurements(_ measurements: [WoundMeasurement]) -> Bool {
        guard let path = measurementsPath else { return false }
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        guard let data = try? encoder.encode(measurements) else { return false }
        return (try? data.write(to: URL(fileURLWithPath: path), options: .atomic)) != nil
    }

    func deleteMeasurementsFile() {
        guard let path = measurementsPath else { return }
        try? FileManager.default.removeItem(atPath: path)
    }
}
