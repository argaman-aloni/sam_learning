from sam_learning.learners import NumericSAMLearner, IncrementalNumericSAMLearner
from sam_learning.learners.baseline_learners.naive_numeric_sam import NaivePolynomialSAMLearning
from sam_learning.learners.baseline_learners.naive_numeric_sam_no_dependency_removal import NaiveNumericSAMLearnerNoDependencyRemoval
from utilities import LearningAlgorithmType

DEFAULT_SPLIT = 5

NUMERIC_ALGORITHMS = [
    LearningAlgorithmType.numeric_sam,
    # LearningAlgorithmType.plan_miner,
    LearningAlgorithmType.polynomial_sam,
    LearningAlgorithmType.raw_numeric_sam,
    LearningAlgorithmType.raw_polynomial_nsam,
    LearningAlgorithmType.naive_nsam,
    LearningAlgorithmType.naive_polysam,
    LearningAlgorithmType.raw_naive_nsam,
    LearningAlgorithmType.raw_naive_polysam,
    LearningAlgorithmType.incremental_nsam,
    LearningAlgorithmType.naive_nsam_no_dependency_removal,
]

DEFAULT_NUMERIC_TOLERANCE = 0.1

MAX_SIZE_MB = 5


NUMERIC_SAM_ALGORITHM_VERSIONS = {
    LearningAlgorithmType.numeric_sam: NumericSAMLearner,
    LearningAlgorithmType.naive_nsam: NaivePolynomialSAMLearning,
    LearningAlgorithmType.incremental_nsam: IncrementalNumericSAMLearner,
    LearningAlgorithmType.naive_nsam_no_dependency_removal: NaiveNumericSAMLearnerNoDependencyRemoval,
}
