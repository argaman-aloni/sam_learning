from utilities import LearningAlgorithmType
from sam_learning.learners import NumericSAMLearner, PolynomialSAMLearning
from sam_learning.learners.naive_numeric_sam import NaiveNumericSAMLearner, NaivePolynomialSAMLearning

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
]

DEFAULT_NUMERIC_TOLERANCE = 0.1

MAX_SIZE_MB = 5


NUMERIC_SAM_ALGORITHM_VERSIONS = {
    LearningAlgorithmType.numeric_sam: NumericSAMLearner,
    LearningAlgorithmType.raw_numeric_sam: NumericSAMLearner,
    LearningAlgorithmType.naive_nsam: NaiveNumericSAMLearner,
    LearningAlgorithmType.polynomial_sam: PolynomialSAMLearning,
    LearningAlgorithmType.raw_polynomial_nsam: PolynomialSAMLearning,
    LearningAlgorithmType.naive_polysam: NaivePolynomialSAMLearning,
    LearningAlgorithmType.raw_naive_nsam: NaiveNumericSAMLearner,
    LearningAlgorithmType.raw_naive_polysam: NaivePolynomialSAMLearning,
}

NO_INSIGHT_NUMERIC_ALGORITHMS = [
    LearningAlgorithmType.raw_numeric_sam.value,
    LearningAlgorithmType.raw_polynomial_nsam.value,
    LearningAlgorithmType.raw_naive_nsam.value,
    LearningAlgorithmType.raw_naive_polysam.value,
]