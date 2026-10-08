//
//  StandardCyborgSDK
//
//
//  SCPointCloudPositions.swift
//  TrueDepthFusion
//
//  Extracts raw vertex positions from SCPointCloud's interleaved point buffer,
//  for use in measurement / hit-testing outside of SceneKit's rendering path.
//
 
import Foundation
import StandardCyborgFusion
import simd
 
extension SCPointCloud {
    /// Returns every point's XYZ position (in the point cloud's native local
    /// coordinate space, typically meters) as a plain Swift array.
    func extractPositions() -> [SIMD3<Float>] {
        let stride = SCPointCloud.pointStride()
        let posOffset = SCPointCloud.positionOffset()
        let count = pointCount
        guard count > 0, stride > 0 else { return [] }
 
        var positions = [SIMD3<Float>](repeating: .zero, count: count)
 
        pointsData.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            guard let base = raw.baseAddress else { return }
            for i in 0..<count {
                let offset = i * stride + posOffset
                var pos = SIMD3<Float>(0, 0, 0)
                withUnsafeMutableBytes(of: &pos) { dst in
                    memcpy(dst.baseAddress!, base + offset, 12) // 3 x Float32
                }
                positions[i] = pos
            }
        }
 
        return positions
    }
}
 
//  Created by Marianny De Leon on 8/20/26.
//  Copyright © 2026 Standard Cyborg. All rights reserved.
//

