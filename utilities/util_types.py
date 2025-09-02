from enum import Enum


class LearningAlgorithmType(Enum):
    sam_learning = 1
    esam_learning = 2
    numeric_sam = 3
    plan_miner = 5
    ma_sam = 7
    conditional_sam = 9
    universal_sam = 10
    online_nsam = 13
    noam_learning = 14
    informative_explorer = 17
    goal_oriented_explorer = 18
    incremental_nsam = 19
    semi_online = 20
    optimistic_explorer = 21
    informative_svm = 22
    ma_sam_plus = 25

    # To remove
    naive_nsam = 15
    naive_polysam = 16


class SolverType(Enum):
    fast_downward = 1
    metric_ff = 2
    enhsp = 3
    fast_forward = 4


class NegativePreconditionPolicy(Enum):
    no_remove = (1,)  # keep all learned preconditions
    soft = (2,)
    hard = (3,)  # no negative preconditions are allowed
    # allow only negative preconditions deducted form cnf solution


NUMERIC_PRECISION = "NUMERIC_PRECISION"
