from sam_learning.learners import NumericSAMLearner, IncrementalNumericSAMLearner, MultiAgentSAM, SAMLearner
from sam_learning.learners.naive_numeric_sam import NaivePolynomialSAMLearning
from utilities import LearningAlgorithmType, NegativePreconditionPolicy

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
]

DEFAULT_NUMERIC_TOLERANCE = 0.1

MAX_SIZE_MB = 5


NUMERIC_SAM_ALGORITHM_VERSIONS = {
    LearningAlgorithmType.numeric_sam: NumericSAMLearner,
    LearningAlgorithmType.naive_nsam: NaivePolynomialSAMLearning,
    LearningAlgorithmType.incremental_nsam: IncrementalNumericSAMLearner,
}

NO_INSIGHT_NUMERIC_ALGORITHMS = [
    LearningAlgorithmType.raw_numeric_sam.value,
    LearningAlgorithmType.raw_polynomial_nsam.value,
    LearningAlgorithmType.raw_naive_nsam.value,
    LearningAlgorithmType.raw_naive_polysam.value,
    LearningAlgorithmType.incremental_nsam.value,
]

SAM_ALGORITHM_VERSIONS = {
    LearningAlgorithmType.sam_learning: SAMLearner,
    LearningAlgorithmType.sam_learning_soft: SAMLearner,
    LearningAlgorithmType.sam_learning_hard: SAMLearner,
}

MA_SAM_ALGORITHM_VERSIONS = {
    LearningAlgorithmType.ma_sam: MultiAgentSAM,
    LearningAlgorithmType.ma_sam_soft: MultiAgentSAM,
    LearningAlgorithmType.ma_sam_hard: MultiAgentSAM,
}

MA_SAM_POLICIES_VERSIONS = {
    LearningAlgorithmType.sam_learning: NegativePreconditionPolicy.no_remove,
    LearningAlgorithmType.sam_learning_soft: NegativePreconditionPolicy.soft,
    LearningAlgorithmType.sam_learning_hard: NegativePreconditionPolicy.hard,
    LearningAlgorithmType.ma_sam: NegativePreconditionPolicy.no_remove,
    LearningAlgorithmType.ma_sam_soft: NegativePreconditionPolicy.soft,
    LearningAlgorithmType.ma_sam_hard: NegativePreconditionPolicy.hard,
}
