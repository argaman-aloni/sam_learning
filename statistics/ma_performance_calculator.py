"""Module responsible for calculating our approach for Multi-Agent precision and recall."""
import csv
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Union
from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import (
    Domain,
    Observation,
    MultiAgentObservation,
)
from statistics import SemanticPerformanceCalculator, _calculate_precision_recall
from utilities import LearningAlgorithmType


class MASamPerformanceCalculator(SemanticPerformanceCalculator):
    """Class responsible for calculating the Multi-Agent precision and recall of a model."""

    def __init__(
        self,
        model_domain: Domain,
        model_domain_path: Path,
        observations: List[Union[Observation, MultiAgentObservation]],
        working_directory_path: Path,
        learning_algorithm: LearningAlgorithmType,
    ):
        super().__init__(model_domain, model_domain_path, observations, working_directory_path, learning_algorithm)
        self.SEMANTIC_PRECISION_STATS = [
            "learning_algorithm",
            "policy",
            "action_name",
            "num_trajectories",
            "precondition_precision",
            "precondition_recall",
            # "effects_precision",
            # "effects_recall",
        ]

    def calculate_preconditions_semantic_performance(
            self, learned_domain: Domain, learned_domain_path: Path
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculates the precision recall values of the learned preconditions.

        :param learned_domain: the action model that was learned using the action model learning algorithm
        :param learned_domain_path: the path to the learned domain.
        :return: the precision and recall dictionaries.
        """
        num_true_positives = defaultdict(int)
        num_false_negatives = defaultdict(int)
        num_false_positives = defaultdict(int)
        self.logger.debug("Starting to calculate the semantic preconditions performance")
        for index, observation in enumerate(self.dataset_observations):
            observation_objects = observation.grounded_objects
            for component in observation.components:
                actions = [component.grounded_action_call] if isinstance(observation, Observation) else component.grounded_joint_action.actions
                for action in actions:
                    if action.name not in learned_domain.actions:
                        continue

                    true_positive, false_positive, false_negative = self._calculate_action_applicability_rate(
                        action, learned_domain_path, component.previous_state, observation_objects,
                    )
                    num_true_positives[action.name] += true_positive
                    num_false_positives[action.name] += false_positive
                    num_false_negatives[action.name] += false_negative

        return _calculate_precision_recall(num_false_negatives, num_false_positives, num_true_positives,
                                           list(learned_domain.actions.keys()))

    def calculate_performance(self, learned_domain_path: Path, num_used_observations: int, policy) -> None:
        """Calculates the model's performance with both the precision and the recall values calculated.
        :param learned_domain_path: the path to the learned action model.
        :param num_used_observations: the number of observations used to learn the action model.
        """
        learned_domain = DomainParser(domain_path=learned_domain_path, partial_parsing=False).parse_domain()
        self.logger.info("Starting to calculate the semantic preconditions performance of the learned domain.")
        preconditions_precision, preconditions_recall = self.calculate_preconditions_semantic_performance(learned_domain, learned_domain_path)
        self.logger.info("Starting to calculate the semantic effects performance of the learned domain.")
        for action_name in self.model_domain.actions:
            action_stats = {
                "learning_algorithm": self.learning_algorithm.name,
                "policy": policy,
                "action_name": action_name,
                "num_trajectories": num_used_observations,
                "precondition_precision": preconditions_precision[action_name],
                "precondition_recall": preconditions_recall[action_name],
            }
            self.combined_stats.append(action_stats)

        self.logger.info(f"Finished calculating the learning performance for {self.learning_algorithm.name}.")