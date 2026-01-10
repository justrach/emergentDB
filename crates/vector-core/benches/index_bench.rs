//! Benchmarks for EmergentDB index implementations.
//!
//! Compares Flat, HNSW, IVF, and PQ indices across different data sizes.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::Rng;
use vector_core::{
    index::{flat::FlatIndex, hnsw::HnswIndex, ivf::IvfIndex, pq::PqIndex, IndexConfig, VectorIndex},
    DistanceMetric, Embedding, NodeId,
};

/// Generate random vectors.
fn generate_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| (0..dim).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect())
        .collect()
}

/// Benchmark insertion performance.
fn bench_insert(c: &mut Criterion) {
    let dim = 128;
    let vectors = generate_vectors(1000, dim);

    let mut group = c.benchmark_group("insert");
    group.throughput(Throughput::Elements(1000));

    // Flat index
    group.bench_function("flat", |b| {
        b.iter(|| {
            let config = IndexConfig::flat(dim, DistanceMetric::Cosine);
            let mut index = FlatIndex::new(config);
            for (i, vec) in vectors.iter().enumerate() {
                index.insert(NodeId::new(i as u64), Embedding::new(vec.clone())).unwrap();
            }
            black_box(index)
        })
    });

    // HNSW index
    group.bench_function("hnsw", |b| {
        b.iter(|| {
            let config = IndexConfig::hnsw(dim, DistanceMetric::Cosine);
            let mut index = HnswIndex::new(config);
            for (i, vec) in vectors.iter().enumerate() {
                index.insert(NodeId::new(i as u64), Embedding::new(vec.clone())).unwrap();
            }
            black_box(index)
        })
    });

    // IVF index (without training - just insertion)
    group.bench_function("ivf_insert_only", |b| {
        b.iter(|| {
            let config = vector_core::index::ivf::IvfConfig::new(dim, 32);
            let mut index = IvfIndex::new(config);
            for (i, vec) in vectors.iter().enumerate() {
                index.insert(NodeId::new(i as u64), Embedding::new(vec.clone())).unwrap();
            }
            black_box(index)
        })
    });

    // PQ index (without training - just insertion)
    group.bench_function("pq_insert_only", |b| {
        b.iter(|| {
            let config = vector_core::index::pq::PqConfig::new(dim, 16).unwrap();
            let mut index = PqIndex::new(config);
            for (i, vec) in vectors.iter().enumerate() {
                index.insert(NodeId::new(i as u64), Embedding::new(vec.clone())).unwrap();
            }
            black_box(index)
        })
    });

    group.finish();
}

/// Benchmark search performance on pre-built indices.
fn bench_search(c: &mut Criterion) {
    let dim = 128;
    let n = 10000;
    let k = 10;
    let vectors = generate_vectors(n, dim);
    let query_vectors = generate_vectors(100, dim);

    let mut group = c.benchmark_group("search");
    group.throughput(Throughput::Elements(100));

    // Build flat index
    let flat_index = {
        let config = IndexConfig::flat(dim, DistanceMetric::Cosine);
        let mut index = FlatIndex::new(config);
        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64), Embedding::new(vec.clone())).unwrap();
        }
        index
    };

    // Build HNSW index
    let hnsw_index = {
        let config = IndexConfig::hnsw(dim, DistanceMetric::Cosine);
        let mut index = HnswIndex::new(config);
        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64), Embedding::new(vec.clone())).unwrap();
        }
        index
    };

    // Build IVF index
    let ivf_index = {
        let config = vector_core::index::ivf::IvfConfig {
            dim,
            metric: DistanceMetric::Cosine,
            num_partitions: 64,
            nprobe: 8,
            kmeans_iterations: 10,
        };
        let mut index = IvfIndex::new(config);
        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64), Embedding::new(vec.clone())).unwrap();
        }
        index.train().unwrap();
        index
    };

    // Build PQ index
    let pq_index = {
        let config = vector_core::index::pq::PqConfig {
            dim,
            num_subvectors: 16,
            num_centroids: 256,
            kmeans_iterations: 10,
        };
        let mut index = PqIndex::new(config);
        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64), Embedding::new(vec.clone())).unwrap();
        }
        index.train().unwrap();
        index
    };

    // Benchmark flat search
    group.bench_function("flat", |b| {
        b.iter(|| {
            for query in &query_vectors {
                let results = flat_index.search(&Embedding::new(query.clone()), k).unwrap();
                black_box(results);
            }
        })
    });

    // Benchmark HNSW search
    group.bench_function("hnsw", |b| {
        b.iter(|| {
            for query in &query_vectors {
                let results = hnsw_index.search(&Embedding::new(query.clone()), k).unwrap();
                black_box(results);
            }
        })
    });

    // Benchmark IVF search
    group.bench_function("ivf", |b| {
        b.iter(|| {
            for query in &query_vectors {
                let results = ivf_index.search(&Embedding::new(query.clone()), k).unwrap();
                black_box(results);
            }
        })
    });

    // Benchmark PQ search
    group.bench_function("pq", |b| {
        b.iter(|| {
            for query in &query_vectors {
                let results = pq_index.search(&Embedding::new(query.clone()), k).unwrap();
                black_box(results);
            }
        })
    });

    group.finish();
}

/// Benchmark training time for IVF and PQ.
fn bench_training(c: &mut Criterion) {
    let dim = 128;
    let vectors = generate_vectors(5000, dim);

    let mut group = c.benchmark_group("training");

    // IVF training
    for &partitions in &[32, 64, 128] {
        group.bench_with_input(
            BenchmarkId::new("ivf", partitions),
            &partitions,
            |b, &partitions| {
                b.iter(|| {
                    let config = vector_core::index::ivf::IvfConfig {
                        dim,
                        metric: DistanceMetric::Cosine,
                        num_partitions: partitions,
                        nprobe: partitions / 4,
                        kmeans_iterations: 10,
                    };
                    let mut index = IvfIndex::new(config);
                    for (i, vec) in vectors.iter().enumerate() {
                        index.insert(NodeId::new(i as u64), Embedding::new(vec.clone())).unwrap();
                    }
                    index.train().unwrap();
                    black_box(index)
                })
            },
        );
    }

    // PQ training
    for &subvectors in &[8, 16, 32] {
        group.bench_with_input(
            BenchmarkId::new("pq", subvectors),
            &subvectors,
            |b, &subvectors| {
                b.iter(|| {
                    let config = vector_core::index::pq::PqConfig {
                        dim,
                        num_subvectors: subvectors,
                        num_centroids: 256,
                        kmeans_iterations: 10,
                    };
                    let mut index = PqIndex::new(config);
                    for (i, vec) in vectors.iter().enumerate() {
                        index.insert(NodeId::new(i as u64), Embedding::new(vec.clone())).unwrap();
                    }
                    index.train().unwrap();
                    black_box(index)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark SIMD distance calculations.
fn bench_distance(c: &mut Criterion) {
    let dim = 1536; // OpenAI embedding dimension
    let mut rng = rand::thread_rng();
    let a: Vec<f32> = (0..dim).map(|_| rng.r#gen::<f32>()).collect();
    let b: Vec<f32> = (0..dim).map(|_| rng.r#gen::<f32>()).collect();

    let mut group = c.benchmark_group("distance");
    group.throughput(Throughput::Elements(1));

    group.bench_function("cosine", |bencher| {
        bencher.iter(|| {
            black_box(vector_core::distance::cosine_similarity_simd(&a, &b))
        })
    });

    group.bench_function("euclidean", |bencher| {
        bencher.iter(|| {
            black_box(vector_core::distance::euclidean_distance_simd(&a, &b))
        })
    });

    group.bench_function("dot_product", |bencher| {
        bencher.iter(|| {
            black_box(vector_core::simd::dot_product_simd(&a, &b))
        })
    });

    group.finish();
}

/// Benchmark different dataset sizes.
fn bench_scalability(c: &mut Criterion) {
    let dim = 128;
    let k = 10;
    let sizes = [1000, 5000, 10000];

    let mut group = c.benchmark_group("scalability");

    for &n in &sizes {
        let vectors = generate_vectors(n, dim);
        let query = generate_vectors(1, dim).pop().unwrap();

        // Pre-build HNSW for this size
        let hnsw_index = {
            let config = IndexConfig::hnsw(dim, DistanceMetric::Cosine);
            let mut index = HnswIndex::new(config);
            for (i, vec) in vectors.iter().enumerate() {
                index.insert(NodeId::new(i as u64), Embedding::new(vec.clone())).unwrap();
            }
            index
        };

        group.bench_with_input(
            BenchmarkId::new("hnsw_search", n),
            &n,
            |b, _| {
                b.iter(|| {
                    let results = hnsw_index.search(&Embedding::new(query.clone()), k).unwrap();
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_distance,
    bench_insert,
    bench_search,
    bench_training,
    bench_scalability,
);
criterion_main!(benches);
