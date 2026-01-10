//! MAP-Elites archive for storing elite solutions in behavior space.

use std::collections::HashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::behavior::BehaviorDescriptor;

/// An individual solution in the archive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Individual<T> {
    /// The solution (e.g., a query string, embedding, etc.)
    pub solution: T,
    /// The behavior descriptor for this solution.
    pub behavior: BehaviorDescriptor,
    /// Fitness score (higher is better).
    pub fitness: f64,
    /// Generation when this individual was added.
    pub generation: usize,
}

impl<T: Clone> Individual<T> {
    /// Create a new individual.
    pub fn new(solution: T, behavior: BehaviorDescriptor, fitness: f64) -> Self {
        Self {
            solution,
            behavior,
            fitness,
            generation: 0,
        }
    }
}

/// Grid cell coordinates in behavior space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CellCoord {
    pub dims: [usize; 4], // Support up to 4 behavior dimensions
}

impl CellCoord {
    pub fn from_behavior(bd: &BehaviorDescriptor, resolution: &[usize]) -> Self {
        let mut dims = [0; 4];
        let features = bd.features();

        for (i, (feat, &res)) in features.iter().zip(resolution.iter()).enumerate() {
            if i >= 4 {
                break;
            }
            // Map [0, 1] range to [0, resolution) cell index
            dims[i] = ((feat * res as f32).floor() as usize).min(res - 1);
        }

        Self { dims }
    }
}

/// MAP-Elites archive with grid-based niching.
pub struct Archive<T> {
    /// Grid cells containing elite individuals.
    cells: RwLock<HashMap<CellCoord, Individual<T>>>,
    /// Resolution per behavior dimension.
    resolution: Vec<usize>,
    /// Total number of evaluations.
    evaluations: RwLock<usize>,
    /// Current generation.
    generation: RwLock<usize>,
}

impl<T: Clone + Send + Sync> Archive<T> {
    /// Create a new archive with the given resolution per dimension.
    ///
    /// # Arguments
    /// * `resolution` - Number of cells per behavior dimension (e.g., [10, 10] for 2D)
    pub fn new(resolution: Vec<usize>) -> Self {
        Self {
            cells: RwLock::new(HashMap::new()),
            resolution,
            evaluations: RwLock::new(0),
            generation: RwLock::new(0),
        }
    }

    /// Try to add an individual to the archive.
    /// Returns true if the individual was added (either new cell or better fitness).
    pub fn try_add(&self, individual: Individual<T>) -> bool {
        let coord = CellCoord::from_behavior(&individual.behavior, &self.resolution);

        let mut cells = self.cells.write();
        *self.evaluations.write() += 1;

        match cells.get(&coord) {
            Some(existing) if existing.fitness >= individual.fitness => false,
            _ => {
                cells.insert(coord, individual);
                true
            }
        }
    }

    /// Get the individual at a specific cell.
    pub fn get(&self, coord: &CellCoord) -> Option<Individual<T>> {
        self.cells.read().get(coord).cloned()
    }

    /// Get all elites in the archive.
    pub fn elites(&self) -> Vec<Individual<T>> {
        self.cells.read().values().cloned().collect()
    }

    /// Number of filled cells.
    pub fn coverage(&self) -> usize {
        self.cells.read().len()
    }

    /// Total possible cells.
    pub fn capacity(&self) -> usize {
        self.resolution.iter().product()
    }

    /// Coverage ratio (0 to 1).
    pub fn coverage_ratio(&self) -> f64 {
        self.coverage() as f64 / self.capacity() as f64
    }

    /// Total evaluations performed.
    pub fn evaluations(&self) -> usize {
        *self.evaluations.read()
    }

    /// Increment generation counter.
    pub fn next_generation(&self) {
        *self.generation.write() += 1;
    }

    /// Current generation.
    pub fn generation(&self) -> usize {
        *self.generation.read()
    }

    /// Get a random elite for mutation.
    pub fn random_elite(&self) -> Option<Individual<T>> {
        use rand::Rng;
        let cells = self.cells.read();
        if cells.is_empty() {
            return None;
        }

        let mut rng = rand::thread_rng();
        let idx = rng.r#gen::<usize>() % cells.len();
        cells.values().nth(idx).cloned()
    }

    /// Get top N elites by fitness.
    pub fn top_elites(&self, n: usize) -> Vec<Individual<T>> {
        let mut elites: Vec<_> = self.cells.read().values().cloned().collect();
        elites.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        elites.truncate(n);
        elites
    }

    /// Statistics about the archive.
    pub fn stats(&self) -> ArchiveStats {
        let cells = self.cells.read();
        let fitnesses: Vec<f64> = cells.values().map(|i| i.fitness).collect();

        ArchiveStats {
            coverage: cells.len(),
            capacity: self.capacity(),
            evaluations: *self.evaluations.read(),
            generation: *self.generation.read(),
            mean_fitness: if fitnesses.is_empty() {
                0.0
            } else {
                fitnesses.iter().sum::<f64>() / fitnesses.len() as f64
            },
            max_fitness: fitnesses.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            min_fitness: fitnesses.iter().copied().fold(f64::INFINITY, f64::min),
        }
    }
}

/// Archive statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveStats {
    pub coverage: usize,
    pub capacity: usize,
    pub evaluations: usize,
    pub generation: usize,
    pub mean_fitness: f64,
    pub max_fitness: f64,
    pub min_fitness: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_archive_add() {
        let archive: Archive<String> = Archive::new(vec![10, 10]);

        let bd = BehaviorDescriptor::new(vec![0.5, 0.5]);
        let ind = Individual::new("test".to_string(), bd, 1.0);

        assert!(archive.try_add(ind.clone()));
        assert_eq!(archive.coverage(), 1);
    }

    #[test]
    fn test_archive_replacement() {
        let archive: Archive<String> = Archive::new(vec![10, 10]);

        let bd = BehaviorDescriptor::new(vec![0.5, 0.5]);
        let ind1 = Individual::new("low".to_string(), bd.clone(), 1.0);
        let ind2 = Individual::new("high".to_string(), bd.clone(), 2.0);
        let ind3 = Individual::new("lower".to_string(), bd, 0.5);

        archive.try_add(ind1);
        assert!(archive.try_add(ind2)); // Should replace
        assert!(!archive.try_add(ind3)); // Should not replace

        let elite = archive.top_elites(1)[0].clone();
        assert_eq!(elite.solution, "high");
    }

    #[test]
    fn test_cell_coord() {
        let bd = BehaviorDescriptor::new(vec![0.25, 0.75]);
        let coord = CellCoord::from_behavior(&bd, &[10, 10]);

        assert_eq!(coord.dims[0], 2); // 0.25 * 10 = 2.5 -> 2
        assert_eq!(coord.dims[1], 7); // 0.75 * 10 = 7.5 -> 7
    }
}
