"""Module responsible for calculating our approach for numeric precision and recall."""
from typing import List

from pddl_plus_parser.models import Domain


class NumericPerformanceCalculator:
    """Class responsible for calculating the numeric precision and recall of a model."""

    sample_points: List[float]
    model_domain: Domain
    learned_domain: Domain
