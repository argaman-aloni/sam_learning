"""Module to contain relevant defect tupes"""
from enum import Enum


class DefectType(Enum):
    """Enum that contains the different defect types."""

    numeric_precondition = 1
    numeric_effect = 2
    removed_predicate = 3
