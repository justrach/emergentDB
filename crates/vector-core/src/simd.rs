//! SIMD primitive operations optimized for Apple M-series chips.
//!
//! This module provides low-level SIMD operations with three tiers:
//! 1. Native ARM NEON intrinsics (best for Apple M1-M4)
//! 2. Portable `wide` crate (fallback for x86_64 AVX2/SSE)
//! 3. Scalar implementation (universal fallback)
//!
//! The implementation automatically selects the best path at compile time.
//!
//! ## Apple M-Series Optimization
//!
//! Apple M-series chips (M1, M2, M3, M4) use ARMv8.4+ with enhanced NEON units:
//! - 128-bit SIMD registers (4x f32 or 2x f64)
//! - Fused multiply-accumulate (FMLA) for dot products
//! - Hardware prefetching for sequential memory access
//!
//! For best performance on M-series, build with:
//! ```bash
//! RUSTFLAGS="-C target-cpu=apple-m1" cargo build --release  # M1/M2
//! RUSTFLAGS="-C target-cpu=native" cargo build --release    # Auto-detect
//! ```

// SIMD width: 4 for ARM NEON (128-bit / 32-bit float), 8 for AVX2
#[cfg(target_arch = "aarch64")]
pub const SIMD_WIDTH: usize = 4;

#[cfg(not(target_arch = "aarch64"))]
pub const SIMD_WIDTH: usize = 8;

// ============================================================================
// ARM NEON Implementation (Apple M-series optimized)
// ============================================================================

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// Dot product using ARM NEON with fused multiply-accumulate.
    /// Optimized for Apple M-series with unrolled loop and prefetching hints.
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let n = a.len();
            let chunks = n / 16; // Process 16 elements per iteration (4x unrolled)
            let remainder_16 = n % 16;

            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();

            // Initialize 4 accumulators for unrolled loop
            let mut sum0 = vdupq_n_f32(0.0);
            let mut sum1 = vdupq_n_f32(0.0);
            let mut sum2 = vdupq_n_f32(0.0);
            let mut sum3 = vdupq_n_f32(0.0);

            // Main loop: process 16 elements per iteration
            for i in 0..chunks {
                let offset = i * 16;

                // Load 4 vectors from each array
                let va0 = vld1q_f32(a_ptr.add(offset));
                let vb0 = vld1q_f32(b_ptr.add(offset));
                let va1 = vld1q_f32(a_ptr.add(offset + 4));
                let vb1 = vld1q_f32(b_ptr.add(offset + 4));
                let va2 = vld1q_f32(a_ptr.add(offset + 8));
                let vb2 = vld1q_f32(b_ptr.add(offset + 8));
                let va3 = vld1q_f32(a_ptr.add(offset + 12));
                let vb3 = vld1q_f32(b_ptr.add(offset + 12));

                // Fused multiply-accumulate (single instruction on M-series)
                sum0 = vfmaq_f32(sum0, va0, vb0);
                sum1 = vfmaq_f32(sum1, va1, vb1);
                sum2 = vfmaq_f32(sum2, va2, vb2);
                sum3 = vfmaq_f32(sum3, va3, vb3);
            }

            // Combine accumulators
            sum0 = vaddq_f32(sum0, sum1);
            sum2 = vaddq_f32(sum2, sum3);
            sum0 = vaddq_f32(sum0, sum2);

            // Handle remaining groups of 4
            let remaining_chunks = remainder_16 / 4;
            let offset = chunks * 16;

            for i in 0..remaining_chunks {
                let idx = offset + i * 4;
                let va = vld1q_f32(a_ptr.add(idx));
                let vb = vld1q_f32(b_ptr.add(idx));
                sum0 = vfmaq_f32(sum0, va, vb);
            }

            // Horizontal sum of final vector
            let result = vaddvq_f32(sum0);

            // Handle final remainder (< 4 elements)
            let final_offset = offset + remaining_chunks * 4;
            let final_remainder = remainder_16 % 4;
            let mut scalar_sum = result;

            for i in 0..final_remainder {
                scalar_sum += a[final_offset + i] * b[final_offset + i];
            }

            scalar_sum
        }
    }

    /// Squared L2 norm using NEON.
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn squared_norm_neon(a: &[f32]) -> f32 {
        unsafe {
            let n = a.len();
            let chunks = n / 4;
            let remainder = n % 4;
            let a_ptr = a.as_ptr();

            let mut sum = vdupq_n_f32(0.0);

            for i in 0..chunks {
                let offset = i * 4;
                let va = vld1q_f32(a_ptr.add(offset));
                sum = vfmaq_f32(sum, va, va);
            }

            let mut result = vaddvq_f32(sum);

            if remainder > 0 {
                let offset = chunks * 4;
                for i in 0..remainder {
                    let val = a[offset + i];
                    result += val * val;
                }
            }

            result
        }
    }

    /// Sum of elements using NEON.
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn sum_neon(a: &[f32]) -> f32 {
        unsafe {
            let n = a.len();
            let chunks = n / 4;
            let remainder = n % 4;
            let a_ptr = a.as_ptr();

            let mut sum = vdupq_n_f32(0.0);

            for i in 0..chunks {
                let offset = i * 4;
                let va = vld1q_f32(a_ptr.add(offset));
                sum = vaddq_f32(sum, va);
            }

            let mut result = vaddvq_f32(sum);

            if remainder > 0 {
                let offset = chunks * 4;
                for i in 0..remainder {
                    result += a[offset + i];
                }
            }

            result
        }
    }

    /// Vector addition using NEON.
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn add_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
        unsafe {
            let n = a.len();
            let chunks = n / 4;
            let remainder = n % 4;

            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            let out_ptr = out.as_mut_ptr();

            for i in 0..chunks {
                let offset = i * 4;
                let va = vld1q_f32(a_ptr.add(offset));
                let vb = vld1q_f32(b_ptr.add(offset));
                let result = vaddq_f32(va, vb);
                vst1q_f32(out_ptr.add(offset), result);
            }

            if remainder > 0 {
                let offset = chunks * 4;
                for i in 0..remainder {
                    out[offset + i] = a[offset + i] + b[offset + i];
                }
            }
        }
    }

    /// Scalar multiplication using NEON.
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn scale_neon(a: &[f32], scalar: f32, out: &mut [f32]) {
        unsafe {
            let n = a.len();
            let chunks = n / 4;
            let remainder = n % 4;

            let a_ptr = a.as_ptr();
            let out_ptr = out.as_mut_ptr();
            let vs = vdupq_n_f32(scalar);

            for i in 0..chunks {
                let offset = i * 4;
                let va = vld1q_f32(a_ptr.add(offset));
                let result = vmulq_f32(va, vs);
                vst1q_f32(out_ptr.add(offset), result);
            }

            if remainder > 0 {
                let offset = chunks * 4;
                for i in 0..remainder {
                    out[offset + i] = a[offset + i] * scalar;
                }
            }
        }
    }

    /// Squared Euclidean distance using NEON.
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn squared_euclidean_neon(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let n = a.len();
            let chunks = n / 4;
            let remainder = n % 4;

            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();

            let mut sum = vdupq_n_f32(0.0);

            for i in 0..chunks {
                let offset = i * 4;
                let va = vld1q_f32(a_ptr.add(offset));
                let vb = vld1q_f32(b_ptr.add(offset));
                let diff = vsubq_f32(va, vb);
                sum = vfmaq_f32(sum, diff, diff);
            }

            let mut result = vaddvq_f32(sum);

            if remainder > 0 {
                let offset = chunks * 4;
                for i in 0..remainder {
                    let diff = a[offset + i] - b[offset + i];
                    result += diff * diff;
                }
            }

            result
        }
    }
}

// ============================================================================
// Portable Implementation (wide crate - x86_64 AVX2/SSE fallback)
// ============================================================================

#[cfg(not(target_arch = "aarch64"))]
mod portable {
    use wide::f32x8;

    const WIDTH: usize = 8;

    #[inline]
    pub fn dot_product_wide(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let chunks = n / WIDTH;
        let remainder = n % WIDTH;

        let mut sum = f32x8::ZERO;

        for i in 0..chunks {
            let offset = i * WIDTH;
            let va = f32x8::from(&a[offset..offset + WIDTH]);
            let vb = f32x8::from(&b[offset..offset + WIDTH]);
            sum += va * vb;
        }

        let mut result = sum.reduce_add();

        if remainder > 0 {
            let offset = chunks * WIDTH;
            for i in 0..remainder {
                result += a[offset + i] * b[offset + i];
            }
        }

        result
    }

    #[inline]
    pub fn squared_norm_wide(a: &[f32]) -> f32 {
        let n = a.len();
        let chunks = n / WIDTH;
        let remainder = n % WIDTH;

        let mut sum = f32x8::ZERO;

        for i in 0..chunks {
            let offset = i * WIDTH;
            let va = f32x8::from(&a[offset..offset + WIDTH]);
            sum += va * va;
        }

        let mut result = sum.reduce_add();

        if remainder > 0 {
            let offset = chunks * WIDTH;
            for i in 0..remainder {
                result += a[offset + i] * a[offset + i];
            }
        }

        result
    }

    #[inline]
    pub fn sum_wide(a: &[f32]) -> f32 {
        let n = a.len();
        let chunks = n / WIDTH;
        let remainder = n % WIDTH;

        let mut sum = f32x8::ZERO;

        for i in 0..chunks {
            let offset = i * WIDTH;
            let va = f32x8::from(&a[offset..offset + WIDTH]);
            sum += va;
        }

        let mut result = sum.reduce_add();

        if remainder > 0 {
            let offset = chunks * WIDTH;
            for i in 0..remainder {
                result += a[offset + i];
            }
        }

        result
    }

    #[inline]
    pub fn add_wide(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len();
        let chunks = n / WIDTH;
        let remainder = n % WIDTH;

        for i in 0..chunks {
            let offset = i * WIDTH;
            let va = f32x8::from(&a[offset..offset + WIDTH]);
            let vb = f32x8::from(&b[offset..offset + WIDTH]);
            let result = va + vb;
            let arr: [f32; 8] = result.into();
            out[offset..offset + WIDTH].copy_from_slice(&arr);
        }

        if remainder > 0 {
            let offset = chunks * WIDTH;
            for i in 0..remainder {
                out[offset + i] = a[offset + i] + b[offset + i];
            }
        }
    }

    #[inline]
    pub fn scale_wide(a: &[f32], scalar: f32, out: &mut [f32]) {
        let n = a.len();
        let chunks = n / WIDTH;
        let remainder = n % WIDTH;
        let vs = f32x8::splat(scalar);

        for i in 0..chunks {
            let offset = i * WIDTH;
            let va = f32x8::from(&a[offset..offset + WIDTH]);
            let result = va * vs;
            let arr: [f32; 8] = result.into();
            out[offset..offset + WIDTH].copy_from_slice(&arr);
        }

        if remainder > 0 {
            let offset = chunks * WIDTH;
            for i in 0..remainder {
                out[offset + i] = a[offset + i] * scalar;
            }
        }
    }

    #[inline]
    pub fn squared_euclidean_wide(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let chunks = n / WIDTH;
        let remainder = n % WIDTH;

        let mut sum = f32x8::ZERO;

        for i in 0..chunks {
            let offset = i * WIDTH;
            let va = f32x8::from(&a[offset..offset + WIDTH]);
            let vb = f32x8::from(&b[offset..offset + WIDTH]);
            let diff = va - vb;
            sum += diff * diff;
        }

        let mut result = sum.reduce_add();

        if remainder > 0 {
            let offset = chunks * WIDTH;
            for i in 0..remainder {
                let diff = a[offset + i] - b[offset + i];
                result += diff * diff;
            }
        }

        result
    }
}

// ============================================================================
// Public API (auto-selects best implementation)
// ============================================================================

/// Compute dot product of two f32 slices using SIMD.
///
/// # Panics
/// Panics if slices have different lengths.
///
/// # Performance
/// - Apple M-series: Uses ARM NEON with FMA (4-wide, ~4x speedup)
/// - x86_64: Uses AVX2/SSE via `wide` crate (8-wide)
#[inline]
pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: We check length equality above, and NEON is always available on aarch64
        unsafe { neon::dot_product_neon(a, b) }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        portable::dot_product_wide(a, b)
    }
}

/// Compute squared L2 norm using SIMD.
#[inline]
pub fn squared_norm_simd(a: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::squared_norm_neon(a) }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        portable::squared_norm_wide(a)
    }
}

/// Compute sum of elements using SIMD.
#[inline]
pub fn sum_simd(a: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::sum_neon(a) }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        portable::sum_wide(a)
    }
}

/// Add two vectors element-wise, storing result in `out`.
#[inline]
pub fn add_simd(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::add_neon(a, b, out) }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        portable::add_wide(a, b, out)
    }
}

/// Multiply vector by scalar using SIMD.
#[inline]
pub fn scale_simd(a: &[f32], scalar: f32, out: &mut [f32]) {
    assert_eq!(a.len(), out.len());

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::scale_neon(a, scalar, out) }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        portable::scale_wide(a, scalar, out)
    }
}

/// Compute squared Euclidean distance between two vectors using SIMD.
#[inline]
pub fn squared_euclidean_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::squared_euclidean_neon(a, b) }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        portable::squared_euclidean_wide(a, b)
    }
}

/// Batch dot product: compute dot product of query against multiple vectors.
///
/// Returns vector of dot products in same order as `vectors`.
#[inline]
pub fn batch_dot_product(query: &[f32], vectors: &[&[f32]]) -> Vec<f32> {
    vectors
        .iter()
        .map(|v| dot_product_simd(query, v))
        .collect()
}

/// Parallel batch dot product using rayon.
pub fn batch_dot_product_parallel(query: &[f32], vectors: &[&[f32]]) -> Vec<f32> {
    use rayon::prelude::*;

    vectors
        .par_iter()
        .map(|v| dot_product_simd(query, v))
        .collect()
}

// ============================================================================
// SIMD-Accelerated Insert Operations (for QD optimization)
// ============================================================================

/// Normalize a vector in-place using SIMD.
/// Returns the original norm for reference.
///
/// This is the core operation for insert optimization - vectors must be
/// normalized before storage for cosine similarity.
#[inline]
pub fn normalize_simd(vec: &mut [f32]) -> f32 {
    let squared_norm = squared_norm_simd(vec);
    let norm = squared_norm.sqrt();

    if norm > 1e-10 {
        let inv_norm = 1.0 / norm;
        // Use scale_simd in-place by creating a temporary
        let n = vec.len();

        #[cfg(target_arch = "aarch64")]
        {
            // Direct in-place scaling for ARM NEON
            let chunks = n / 4;
            let remainder = n % 4;

            unsafe {
                use std::arch::aarch64::*;
                let vs = vdupq_n_f32(inv_norm);
                let ptr = vec.as_mut_ptr();

                for i in 0..chunks {
                    let offset = i * 4;
                    let v = vld1q_f32(ptr.add(offset));
                    let scaled = vmulq_f32(v, vs);
                    vst1q_f32(ptr.add(offset), scaled);
                }

                let offset = chunks * 4;
                for i in 0..remainder {
                    vec[offset + i] *= inv_norm;
                }
            }
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            use wide::f32x8;
            const WIDTH: usize = 8;

            let chunks = n / WIDTH;
            let remainder = n % WIDTH;
            let vs = f32x8::splat(inv_norm);

            for i in 0..chunks {
                let offset = i * WIDTH;
                let v = f32x8::from(&vec[offset..offset + WIDTH]);
                let scaled = v * vs;
                let arr: [f32; 8] = scaled.into();
                vec[offset..offset + WIDTH].copy_from_slice(&arr);
            }

            let offset = chunks * WIDTH;
            for i in 0..remainder {
                vec[offset + i] *= inv_norm;
            }
        }
    }

    norm
}

/// Batch normalize multiple vectors using SIMD.
/// Returns vector of original norms.
///
/// This is optimized for bulk insert operations where many vectors
/// need normalization before storage.
#[inline]
pub fn batch_normalize_simd(vectors: &mut [Vec<f32>]) -> Vec<f32> {
    vectors
        .iter_mut()
        .map(|v| normalize_simd(v))
        .collect()
}

/// Parallel batch normalize using rayon + SIMD.
/// Best for large batch inserts (100+ vectors).
pub fn batch_normalize_parallel(vectors: &mut [Vec<f32>]) -> Vec<f32> {
    use rayon::prelude::*;

    vectors
        .par_iter_mut()
        .map(|v| normalize_simd(v))
        .collect()
}

/// Cache-friendly chunked normalization with prefetching hints.
/// Processes vectors in L2 cache-sized chunks for better locality.
pub fn batch_normalize_chunked(vectors: &mut [Vec<f32>]) -> Vec<f32> {
    // L2 cache is ~256KB on M-series, process in ~64KB chunks
    const CHUNK_SIZE: usize = 64;

    let mut norms = Vec::with_capacity(vectors.len());

    for chunk in vectors.chunks_mut(CHUNK_SIZE) {
        // Process chunk - all vectors fit in L2 cache
        for v in chunk.iter_mut() {
            norms.push(normalize_simd(v));
        }
    }

    norms
}

/// Unrolled normalization - process 4 vectors per iteration for CPU pipelining.
/// Exploits instruction-level parallelism on superscalar CPUs.
pub fn batch_normalize_unrolled(vectors: &mut [Vec<f32>]) -> Vec<f32> {
    let n = vectors.len();
    let mut norms = Vec::with_capacity(n);

    // Process 4 vectors at a time for better pipelining
    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let base = i * 4;
        // Compute all 4 squared norms first (data independent)
        let sq0 = squared_norm_simd(&vectors[base]);
        let sq1 = squared_norm_simd(&vectors[base + 1]);
        let sq2 = squared_norm_simd(&vectors[base + 2]);
        let sq3 = squared_norm_simd(&vectors[base + 3]);

        // Then normalize all 4 (can overlap with next iteration's loads)
        let n0 = sq0.sqrt();
        let n1 = sq1.sqrt();
        let n2 = sq2.sqrt();
        let n3 = sq3.sqrt();

        if n0 > 1e-10 { scale_in_place(&mut vectors[base], 1.0 / n0); }
        if n1 > 1e-10 { scale_in_place(&mut vectors[base + 1], 1.0 / n1); }
        if n2 > 1e-10 { scale_in_place(&mut vectors[base + 2], 1.0 / n2); }
        if n3 > 1e-10 { scale_in_place(&mut vectors[base + 3], 1.0 / n3); }

        norms.extend([n0, n1, n2, n3]);
    }

    // Handle remainder
    for i in (chunks * 4)..n {
        norms.push(normalize_simd(&mut vectors[i]));
    }

    norms
}

/// Interleaved normalization - compute norms for multiple vectors before scaling.
/// Better memory bandwidth utilization.
pub fn batch_normalize_interleaved(vectors: &mut [Vec<f32>]) -> Vec<f32> {
    let n = vectors.len();

    // First pass: compute all norms (read-only, cache friendly)
    let norms: Vec<f32> = vectors
        .iter()
        .map(|v| squared_norm_simd(v).sqrt())
        .collect();

    // Second pass: scale all vectors (write, sequential)
    for (v, &norm) in vectors.iter_mut().zip(norms.iter()) {
        if norm > 1e-10 {
            scale_in_place(v, 1.0 / norm);
        }
    }

    norms
}

/// Scale vector in-place by scalar using SIMD.
#[inline]
fn scale_in_place(vec: &mut [f32], scale: f32) {
    let n = vec.len();

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;

        let vs = vdupq_n_f32(scale);
        let chunks = n / 4;

        for i in 0..chunks {
            let offset = i * 4;
            let v = vld1q_f32(vec.as_ptr().add(offset));
            let scaled = vmulq_f32(v, vs);
            vst1q_f32(vec.as_mut_ptr().add(offset), scaled);
        }

        // Scalar remainder
        for i in (chunks * 4)..n {
            vec[i] *= scale;
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        use wide::f32x8;
        const WIDTH: usize = 8;

        let chunks = n / WIDTH;
        let vs = f32x8::splat(scale);

        for i in 0..chunks {
            let offset = i * WIDTH;
            let v = f32x8::from(&vec[offset..offset + WIDTH]);
            let scaled = v * vs;
            let arr: [f32; 8] = scaled.into();
            vec[offset..offset + WIDTH].copy_from_slice(&arr);
        }

        for i in (chunks * WIDTH)..n {
            vec[i] *= scale;
        }
    }
}

/// Insert strategy enum for QD optimization.
/// The MAP-Elites genome can evolve which strategy works best.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InsertStrategy {
    /// Sequential SIMD normalization + insert
    SimdSequential,
    /// Batch SIMD normalization, then sequential insert
    SimdBatch,
    /// Parallel SIMD normalization + parallel insert (rayon)
    SimdParallel,
    /// Pre-normalized vectors (skip normalization)
    PreNormalized,
    /// Cache-friendly chunked processing with prefetching
    SimdChunked,
    /// Unrolled loop for better CPU pipelining
    SimdUnrolled,
    /// Process multiple vectors interleaved for cache efficiency
    SimdInterleaved,
}

impl Default for InsertStrategy {
    fn default() -> Self {
        InsertStrategy::SimdSequential
    }
}

impl std::fmt::Display for InsertStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InsertStrategy::SimdSequential => write!(f, "SIMD-Seq"),
            InsertStrategy::SimdBatch => write!(f, "SIMD-Batch"),
            InsertStrategy::SimdParallel => write!(f, "SIMD-Parallel"),
            InsertStrategy::PreNormalized => write!(f, "PreNorm"),
            InsertStrategy::SimdChunked => write!(f, "SIMD-Chunked"),
            InsertStrategy::SimdUnrolled => write!(f, "SIMD-Unrolled"),
            InsertStrategy::SimdInterleaved => write!(f, "SIMD-Interleaved"),
        }
    }
}

/// Benchmark insert strategies and return throughput (vectors/sec).
pub fn benchmark_insert_strategy(
    strategy: InsertStrategy,
    vectors: &[Vec<f32>],
) -> f32 {
    use std::time::Instant;

    // Clone for fair comparison (each strategy modifies vectors)
    let mut test_vectors: Vec<Vec<f32>> = vectors.iter().cloned().collect();

    let start = Instant::now();

    match strategy {
        InsertStrategy::SimdSequential => {
            for v in test_vectors.iter_mut() {
                normalize_simd(v);
            }
        }
        InsertStrategy::SimdBatch => {
            batch_normalize_simd(&mut test_vectors);
        }
        InsertStrategy::SimdParallel => {
            batch_normalize_parallel(&mut test_vectors);
        }
        InsertStrategy::PreNormalized => {
            // No-op: assumes vectors are already normalized
        }
        InsertStrategy::SimdChunked => {
            batch_normalize_chunked(&mut test_vectors);
        }
        InsertStrategy::SimdUnrolled => {
            batch_normalize_unrolled(&mut test_vectors);
        }
        InsertStrategy::SimdInterleaved => {
            batch_normalize_interleaved(&mut test_vectors);
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let throughput = vectors.len() as f64 / elapsed;

    throughput as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product_simd() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let b = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let result = dot_product_simd(&a, &b);
        assert!((result - 55.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_aligned() {
        // Exactly aligned to SIMD width
        let a = vec![1.0; 16];
        let b = vec![2.0; 16];

        let result = dot_product_simd(&a, &b);
        assert!((result - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_squared_norm_simd() {
        let a = vec![3.0, 4.0];
        let result = squared_norm_simd(&a);
        assert!((result - 25.0).abs() < 1e-6);
    }

    #[test]
    fn test_sum_simd() {
        let a: Vec<f32> = (1..=100).map(|x| x as f32).collect();
        let result = sum_simd(&a);
        let expected = 100.0 * 101.0 / 2.0;
        assert!((result - expected).abs() < 1e-3);
    }

    #[test]
    fn test_add_simd() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let mut out = vec![0.0; 5];

        add_simd(&a, &b, &mut out);

        assert_eq!(out, vec![11.0, 22.0, 33.0, 44.0, 55.0]);
    }

    #[test]
    fn test_scale_simd() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut out = vec![0.0; 9];

        scale_simd(&a, 2.0, &mut out);

        let expected: Vec<f32> = a.iter().map(|x| x * 2.0).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn test_squared_euclidean_simd() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        // (4-1)^2 + (5-2)^2 + (6-3)^2 = 9 + 9 + 9 = 27
        let result = squared_euclidean_simd(&a, &b);
        assert!((result - 27.0).abs() < 1e-6);
    }

    #[test]
    fn test_large_vector() {
        // Test with typical embedding sizes
        let dim = 1536; // OpenAI embedding dimension
        let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..dim).map(|i| ((dim - i) as f32) * 0.001).collect();

        let result = dot_product_simd(&a, &b);

        // Verify against scalar computation
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!((result - expected).abs() < 0.01);
    }

    #[test]
    fn test_empty_vector() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let result = dot_product_simd(&a, &b);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_single_element() {
        let a = vec![3.0];
        let b = vec![4.0];
        let result = dot_product_simd(&a, &b);
        assert!((result - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_simd() {
        let mut vec = vec![3.0, 4.0];
        let norm = normalize_simd(&mut vec);

        assert!((norm - 5.0).abs() < 1e-6);
        assert!((vec[0] - 0.6).abs() < 1e-6);
        assert!((vec[1] - 0.8).abs() < 1e-6);

        // Verify normalized
        let new_norm = squared_norm_simd(&vec).sqrt();
        assert!((new_norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_simd_large() {
        // Test with typical embedding size
        let dim = 768;
        let mut vec: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01).collect();
        let original_norm = squared_norm_simd(&vec).sqrt();

        let returned_norm = normalize_simd(&mut vec);

        assert!((returned_norm - original_norm).abs() < 1e-3);

        // Verify normalized
        let new_norm = squared_norm_simd(&vec).sqrt();
        assert!((new_norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_batch_normalize_simd() {
        let mut vectors = vec![
            vec![3.0, 4.0],
            vec![1.0, 0.0],
            vec![0.0, 0.0, 5.0],
        ];

        let norms = batch_normalize_simd(&mut vectors);

        assert!((norms[0] - 5.0).abs() < 1e-6);
        assert!((norms[1] - 1.0).abs() < 1e-6);
        assert!((norms[2] - 5.0).abs() < 1e-6);

        // Verify all normalized
        for v in &vectors {
            let norm = squared_norm_simd(v).sqrt();
            assert!((norm - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_insert_strategy_benchmark() {
        let dim = 768;
        let n = 100;
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|i| (i as f32) * 0.001).collect())
            .collect();

        let throughput = benchmark_insert_strategy(InsertStrategy::SimdSequential, &vectors);
        assert!(throughput > 0.0); // Should complete without error

        let throughput_batch = benchmark_insert_strategy(InsertStrategy::SimdBatch, &vectors);
        assert!(throughput_batch > 0.0);

        let throughput_parallel = benchmark_insert_strategy(InsertStrategy::SimdParallel, &vectors);
        assert!(throughput_parallel > 0.0);
    }
}
