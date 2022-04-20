from enum import Enum


class LearningAlgorithmType(Enum):
    sam_learning = 1
    esam_learning = 2
    numeric_sam = 3


class SolverType(Enum):
    fast_downward = 1
    metric_ff = 2