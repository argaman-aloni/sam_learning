from enum import Enum


class LearningAlgorithmType(Enum):
    sam_learning = 1
    esam_learning = 2
    numeric_sam = 3
    plan_miner = 4
    polynomial_sam = 5


class SolverType(Enum):
    fast_downward = 1
    metric_ff = 2
    enhsp = 3


class SolutionOutputTypes(Enum):
    ok = 1
    no_solution = 2
    timeout = 3
    not_applicable = 4