"""Module to contain relevant defect tupes"""
from enum import Enum


class DefectType(Enum):
    """Enum that contains the different defect types."""

    numeric_precondition_sign = 1
    numeric_precondition_numeric_change = 2
    numeric_effect = 3
    removed_predicate = 4
    removed_numeric_precondition = 5


class RepairAlgorithmType(Enum):

    numeric_sam = 1
    oblique_tree = 2
    # extended_svc = 3