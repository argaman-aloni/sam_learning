from enum import Enum


class LearningAlgorithmType(Enum):
    sam_learning = 1
    esam_learning = 2
    numeric_sam = 3
    raw_numeric_sam = 4
    plan_miner = 5
    polynomial_sam = 6
    ma_sam = 7
    ma_sam_baseline = 8
    # learning algorithms relating to fault repair not to be used for POL
    oblique_tree = 9
    extended_svc = 10


class SolverType(Enum):
    fast_downward = 1
    metric_ff = 2
    enhsp = 3


class SolutionOutputTypes(Enum):
    ok = 1
    no_solution = 2
    timeout = 3
    not_applicable = 4
    goal_not_achieved = 5
