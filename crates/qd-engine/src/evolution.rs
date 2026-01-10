//! MAP-Elites evolution engine.

use rayon::prelude::*;

use crate::archive::{Archive, Individual};
use crate::behavior::{BehaviorCharacterizer, BehaviorDescriptor};

/// Configuration for MAP-Elites.
#[derive(Debug, Clone)]
pub struct MapElitesConfig {
    /// Grid resolution per behavior dimension.
    pub resolution: Vec<usize>,
    /// Number of initial random solutions.
    pub initial_population: usize,
    /// Batch size for parallel evaluation.
    pub batch_size: usize,
    /// Maximum generations.
    pub max_generations: usize,
    /// Early stopping if coverage reaches this ratio.
    pub target_coverage: f64,
}

impl Default for MapElitesConfig {
    fn default() -> Self {
        Self {
            resolution: vec![20, 20],
            initial_population: 100,
            batch_size: 50,
            max_generations: 1000,
            target_coverage: 0.8,
        }
    }
}

/// Trait for solution generators.
pub trait SolutionGenerator<T>: Send + Sync {
    /// Generate a random solution.
    fn random(&self) -> T;

    /// Mutate an existing solution.
    fn mutate(&self, solution: &T) -> T;
}

/// Trait for fitness evaluators.
pub trait FitnessEvaluator<T>: Send + Sync {
    /// Evaluate the fitness of a solution.
    fn evaluate(&self, solution: &T) -> f64;
}

/// MAP-Elites algorithm implementation.
pub struct MapElites<T, G, F, C>
where
    T: Clone + Send + Sync,
    G: SolutionGenerator<T>,
    F: FitnessEvaluator<T>,
    C: BehaviorCharacterizer<T>,
{
    config: MapElitesConfig,
    archive: Archive<T>,
    generator: G,
    evaluator: F,
    characterizer: C,
}

impl<T, G, F, C> MapElites<T, G, F, C>
where
    T: Clone + Send + Sync,
    G: SolutionGenerator<T>,
    F: FitnessEvaluator<T>,
    C: BehaviorCharacterizer<T> + Sync,
{
    /// Create a new MAP-Elites instance.
    pub fn new(config: MapElitesConfig, generator: G, evaluator: F, characterizer: C) -> Self {
        let archive = Archive::new(config.resolution.clone());

        Self {
            config,
            archive,
            generator,
            evaluator,
            characterizer,
        }
    }

    /// Initialize the archive with random solutions.
    pub fn initialize(&self) {
        let solutions: Vec<T> = (0..self.config.initial_population)
            .into_par_iter()
            .map(|_| self.generator.random())
            .collect();

        for solution in solutions {
            let fitness = self.evaluator.evaluate(&solution);
            let behavior = self.characterizer.characterize(&solution);
            let individual = Individual::new(solution, behavior, fitness);
            self.archive.try_add(individual);
        }
    }

    /// Run one generation of evolution.
    pub fn step(&self) -> bool {
        let batch_size = self.config.batch_size;

        // Generate offspring by mutating random elites
        let offspring: Vec<T> = (0..batch_size)
            .into_par_iter()
            .filter_map(|_| {
                self.archive.random_elite().map(|parent| {
                    self.generator.mutate(&parent.solution)
                })
            })
            .collect();

        if offspring.is_empty() {
            return false;
        }

        // Evaluate and try to add to archive
        let mut added = 0;
        for solution in offspring {
            let fitness = self.evaluator.evaluate(&solution);
            let behavior = self.characterizer.characterize(&solution);
            let individual = Individual::new(solution, behavior, fitness);

            if self.archive.try_add(individual) {
                added += 1;
            }
        }

        self.archive.next_generation();

        added > 0
    }

    /// Run evolution until termination condition.
    pub fn run(&self) -> EvolutionResult<T> {
        self.initialize();

        let mut stagnation = 0;
        let max_stagnation = 50; // Stop if no progress for 50 generations

        for _ in 0..self.config.max_generations {
            let added = self.step();

            if !added {
                stagnation += 1;
                if stagnation >= max_stagnation {
                    break;
                }
            } else {
                stagnation = 0;
            }

            // Check target coverage
            if self.archive.coverage_ratio() >= self.config.target_coverage {
                break;
            }
        }

        EvolutionResult {
            elites: self.archive.elites(),
            stats: self.archive.stats(),
        }
    }

    /// Get the current archive.
    pub fn archive(&self) -> &Archive<T> {
        &self.archive
    }

    /// Get top N diverse solutions.
    pub fn diverse_solutions(&self, n: usize) -> Vec<T> {
        self.archive.top_elites(n).into_iter().map(|i| i.solution).collect()
    }
}

/// Result of an evolution run.
pub struct EvolutionResult<T> {
    pub elites: Vec<Individual<T>>,
    pub stats: crate::archive::ArchiveStats,
}

// ============================================================================
// Example implementations for string queries
// ============================================================================

/// Simple string mutation generator.
pub struct StringMutationGenerator {
    base_queries: Vec<String>,
    mutation_rate: f64,
}

impl StringMutationGenerator {
    pub fn new(base_queries: Vec<String>, mutation_rate: f64) -> Self {
        Self {
            base_queries,
            mutation_rate,
        }
    }
}

impl SolutionGenerator<String> for StringMutationGenerator {
    fn random(&self) -> String {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let idx = rng.r#gen::<usize>() % self.base_queries.len();
        self.base_queries[idx].clone()
    }

    fn mutate(&self, solution: &String) -> String {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Simple word-level mutation
        let words: Vec<&str> = solution.split_whitespace().collect();
        if words.is_empty() {
            return self.random();
        }

        let mut new_words: Vec<String> = words.iter().map(|w| w.to_string()).collect();

        // Add, remove, or replace a word
        let mutation_type = rng.r#gen::<u8>() % 3;
        match mutation_type {
            0 if new_words.len() > 1 => {
                // Remove random word
                let idx = rng.r#gen::<usize>() % new_words.len();
                new_words.remove(idx);
            }
            1 => {
                // Duplicate random word
                let idx = rng.r#gen::<usize>() % new_words.len();
                let word = new_words[idx].clone();
                new_words.push(word);
            }
            _ => {
                // Shuffle words
                let idx1 = rng.r#gen::<usize>() % new_words.len();
                let idx2 = rng.r#gen::<usize>() % new_words.len();
                new_words.swap(idx1, idx2);
            }
        }

        new_words.join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::behavior::TextBehaviorCharacterizer;

    struct MockEvaluator;

    impl FitnessEvaluator<String> for MockEvaluator {
        fn evaluate(&self, solution: &String) -> f64 {
            solution.len() as f64
        }
    }

    #[test]
    fn test_map_elites_basic() {
        let config = MapElitesConfig {
            resolution: vec![5, 5],
            initial_population: 10,
            batch_size: 5,
            max_generations: 10,
            target_coverage: 0.5,
        };

        let generator = StringMutationGenerator::new(
            vec![
                "fast query".to_string(),
                "accurate search".to_string(),
                "quick lookup".to_string(),
            ],
            0.3,
        );

        let evaluator = MockEvaluator;

        let characterizer = TextBehaviorCharacterizer::new(vec![
            vec!["fast".to_string(), "quick".to_string()],
            vec!["accurate".to_string(), "precise".to_string()],
        ]);

        let me = MapElites::new(config, generator, evaluator, characterizer);
        let result = me.run();

        assert!(!result.elites.is_empty());
        assert!(result.stats.evaluations > 0);
    }
}
