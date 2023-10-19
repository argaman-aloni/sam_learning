"""Module responsible for calculating our approach for numeric precision and recall."""
import csv
import logging
import random
from collections import defaultdict
from itertools import permutations
from pathlib import Path
from typing import List, Dict, Tuple, Any, Union

from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain, Observation, ActionCall, State, MultiAgentObservation, \
    JointActionCall, Operator, PDDLObject, PDDLType, Action

from sam_learning.core import VocabularyCreator
from statistics.performance_calculation_utils import _calculate_single_action_applicability_rate
from utilities import LearningAlgorithmType

SEMANTIC_PRECISION_STATS = ["action_name", "num_trajectories", "precondition_precision", "precondition_recall",
                            "effects_precision", "effects_recall"]


def _calculate_precision_recall(
        num_false_negatives: Dict[str, int],
        num_false_positives: Dict[str, int], num_true_positives: Dict[str, int]) -> Tuple[
    Dict[str, float], Dict[str, float]]:
    """Calculates the precision and recall values for each action.

    :param num_false_negatives: the number of false negatives for each action.
    :param num_false_positives: the number of false positives for each action.
    :param num_true_positives: the number of true positives for each action.
    :return: a tuple of two dictionaries, one for the precision values and one for the recall values.
    """
    precision_dict = {}
    recall_dict = {}
    for action_name, tp_rate in num_true_positives.items():
        if tp_rate == 0 and num_false_positives[action_name] == 0:
            precision_dict[action_name] = 1

        if tp_rate == 0 and num_false_negatives[action_name] == 0:
            precision_dict[action_name] = 1

        if tp_rate == 0:
            precision_dict[action_name] = 0
            recall_dict[action_name] = 0
            continue

        precision_dict[action_name] = tp_rate / (tp_rate + num_false_positives[action_name])
        recall_dict[action_name] = tp_rate / (tp_rate + num_false_negatives[action_name])
    return precision_dict, recall_dict


def choose_objects_subset(array: List[str], subset_size: int) -> List[Tuple[str]]:
    """Choose r items our of a list size n.

    :param array: the input list.
    :param subset_size: the size of the subset.
    :return: a list containing subsets of the original list.
    """
    return list(permutations(array, subset_size))


class SemanticPerformanceCalculator:
    """Class responsible for calculating the semantic precision and recall of a model."""

    model_domain: Domain
    dataset_observations: List[Observation]
    learning_algorithm: LearningAlgorithmType
    combined_stats: List[Dict[str, Any]]
    logger: logging.Logger
    results_dir_path: Path

    def __init__(self, model_domain: Domain, observations: List[Union[Observation, MultiAgentObservation]],
                 working_directory_path: Path, learning_algorithm: LearningAlgorithmType):
        self.logger = logging.getLogger(__name__)
        self.model_domain = model_domain
        self.dataset_observations = observations
        self.learning_algorithm = learning_algorithm
        self.combined_stats = []
        self.results_dir_path = working_directory_path / "results_directory"
        self.vocabulary_creator = VocabularyCreator()

    def _validate_type_matching(self, grounded_signatures: Dict[str, PDDLType], action: Action) -> bool:
        """Validates that the types of the grounded signature match the types of the predicate signature.

        :param grounded_signatures: the grounded predicate signature.
        :param action: the lifted action.
        :return: whether the types match.
        """
        for object_name, predicate_parameter in zip(grounded_signatures, action.signature):
            parameter_type = action.signature[predicate_parameter]
            grounded_type = grounded_signatures[object_name]
            if not grounded_type.is_sub_type(parameter_type):
                self.logger.debug(f"The combination of objects - {grounded_signatures}"
                                  f" does not fit {action.name}'s signature")
                return False

        return True

    def _create_grounded_action_vocabulary(
            self, domain: Domain, observed_objects: Dict[str, PDDLObject]) -> List[ActionCall]:
        """Create a vocabulary of random combinations of the predicates parameters and objects.

        :param domain: the domain containing the predicates and the action signatures.
        :param observed_objects: the objects that were observed in the trajectory.
        :return: list containing all the predicates with the different combinations of parameters.
        """
        self.logger.info("Creating grounded action vocabulary with sampled ground actions")
        possible_ground_actions = self.vocabulary_creator.create_grounded_actions_vocabulary(
            domain=self.model_domain, observed_objects=observed_objects)

        vocabulary = []
        for action in domain.actions.values():
            self.logger.debug(f"Creating grounded action vocabulary for action {action.name}")
            action_vocabulary = [action_call for action_call in possible_ground_actions if
                                 action_call.name == action.name]
            sampled_action_vocabulary = random.choices(action_vocabulary, k=10)
            vocabulary.extend(sampled_action_vocabulary)

        return vocabulary

    def _calculate_action_applicability_rate(
            self, action_call: Union[ActionCall, JointActionCall], learned_domain: Domain,
            num_false_negatives: Dict[str, int],
            num_false_positives: Dict[str, int], num_true_positives: Dict[str, int], observed_state: State,
            problem_objects: Dict[str, PDDLObject]) -> None:
        """Test whether an action is applicable in both the model domain and the generated domain.

        :param action_call: the action call that is tested for applicability.
        :param learned_domain: the domain that was learned using the action model learning algorithm.
        :param num_false_negatives: the dictionary mapping between the action name and the number of false negative
            executions.
        :param num_false_positives: the dictionary mapping between the action name and the number of false positive
            executions.
        :param num_true_positives: the dictionary mapping between the action name and the number of true positive
            executions.
        :param observed_state: the state that is currently being tested.
        """
        if isinstance(action_call, JointActionCall):
            for action in action_call.actions:
                _calculate_single_action_applicability_rate(
                    action, learned_domain, self.model_domain, num_false_negatives, num_false_positives,
                    num_true_positives, observed_state, problem_objects)

            return

        if action_call.name not in learned_domain.actions:
            num_false_negatives[action_call.name] += 1
            num_false_positives[action_call.name] += 0
            num_true_positives[action_call.name] += 0
            return

        _calculate_single_action_applicability_rate(
            action_call, learned_domain, self.model_domain, num_false_negatives, num_false_positives,
            num_true_positives, observed_state, problem_objects)

    def _calculate_effects_difference_rate(
            self, observation: Observation, learned_domain: Domain,
            num_false_negatives: Dict[str, int], num_false_positives: Dict[str, int],
            num_true_positives: Dict[str, int]) -> None:
        """Calculates the effects difference rate for each action.

        :param observation: the observation that is being tested.
        :param learned_domain: the domain that was learned using the action model learning algorithm.
        :param num_false_negatives: the dictionary mapping between the action name and the number of false negative
        :param num_false_positives: the dictionary mapping between the action name and the number of false positive
        :param num_true_positives: the dictionary mapping between the action name and the number of true positive
        """
        self.logger.info("Calculating effects difference rate")
        for observation_triplet in observation.components:
            model_previous_state = observation_triplet.previous_state
            executed_action = observation_triplet.grounded_action_call
            model_next_state = observation_triplet.next_state
            if executed_action.name not in learned_domain.actions:
                continue

            try:
                learned_operator = Operator(
                    action=learned_domain.actions[executed_action.name], domain=learned_domain,
                    grounded_action_call=executed_action.parameters, problem_objects=observation.grounded_objects)
                learned_next_state = learned_operator.apply(model_previous_state)

                self.logger.debug("Validating if there are any false negatives.")
                for lifted_predicate in model_next_state.state_predicates:
                    if lifted_predicate not in learned_next_state.state_predicates:
                        num_false_negatives[executed_action.name] += 1
                        break

                    for grounded_predicate in model_next_state.state_predicates[lifted_predicate]:
                        if grounded_predicate.untyped_representation not in learned_next_state.serialize():
                            num_false_negatives[executed_action.name] += 1
                            break

                        num_true_positives[executed_action.name] += 1

                self.logger.debug("Trying to validate that there are no false positives.")
                for lifted_predicate in learned_next_state.state_predicates:
                    if len(learned_next_state.state_predicates[lifted_predicate]) == 0:
                        continue

                    if lifted_predicate not in model_next_state.state_predicates:
                        num_false_positives[executed_action.name] += 1
                        break

                    for grounded_predicate in learned_next_state.state_predicates[lifted_predicate]:
                        if grounded_predicate.untyped_representation not in model_next_state.serialize():
                            num_false_positives[executed_action.name] += 1
                            break

            except ValueError:
                self.logger.debug("The action is not applicable in the state.")
                continue

    def calculate_preconditions_semantic_performance(
            self, learned_domain: Domain) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculates the precision recall values of the learned preconditions.

        :param learned_domain: the action model that was learned using the action model learning algorithm
        :return: the precision and recall dictionaries.
        """
        num_true_positives = defaultdict(int)
        num_false_negatives = defaultdict(int)
        num_false_positives = defaultdict(int)
        for observation in self.dataset_observations:
            observation_objects = observation.grounded_objects
            possible_ground_actions = self._create_grounded_action_vocabulary(self.model_domain, observation_objects)
            for component in observation.components:
                tested_state = component.previous_state
                possible_ground_actions.append(component.grounded_action_call)
                for action in possible_ground_actions:
                    if action.name not in learned_domain.actions:
                        continue

                    self._calculate_action_applicability_rate(
                        action, learned_domain, num_false_negatives, num_false_positives, num_true_positives,
                        tested_state, observation_objects)

        return _calculate_precision_recall(num_false_negatives, num_false_positives, num_true_positives)

    def calculate_effects_semantic_performance(
            self, learned_domain: Domain) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculates the precision recall values of the learned effects.

        :param learned_domain: the action model that was learned using the action model learning algorithm
        :return: the precision and recall dictionaries.
        """
        self.logger.info("Starting to calculate the semantic effects performance")
        num_true_positives = defaultdict(int)
        num_false_negatives = defaultdict(int)
        num_false_positives = defaultdict(int)
        for observation in self.dataset_observations:
            self._calculate_effects_difference_rate(
                observation, learned_domain, num_false_negatives, num_false_positives, num_true_positives)

        return _calculate_precision_recall(num_false_negatives, num_false_positives, num_true_positives)

    def calculate_performance(self, learned_domain_path: Path, num_used_observations: int) -> None:
        """Calculate the semantic precision and recall of the learned domain.

        :param learned_domain_path: the path to the learned domain.
        :param num_used_observations: the number of observations used to learn the domain.
        """
        learned_domain = DomainParser(domain_path=learned_domain_path, partial_parsing=False).parse_domain()
        self.logger.info("Starting to calculate the semantic preconditions performance of the learned domain.")
        preconditions_precision, preconditions_recall = self.calculate_preconditions_semantic_performance(
            learned_domain)
        self.logger.info("Starting to calculate the semantic effects performance of the learned domain.")
        effects_precision, effects_recall = self.calculate_effects_semantic_performance(learned_domain)
        for action_name in self.model_domain.actions:
            action_stats = {
                "action_name": action_name,
                "num_trajectories": num_used_observations,
                "precondition_precision": preconditions_precision.get(action_name, 0),
                "precondition_recall": preconditions_recall.get(action_name, 0),
                "effects_precision": effects_precision.get(action_name, 0),
                "effects_recall": effects_recall.get(action_name, 0)
            }
            self.combined_stats.append(action_stats)

    def export_semantic_performance(self, fold_num: int) -> None:
        """Exports the precision values of the learned preconditions to a CSV file."""
        statistics_path = self.results_dir_path / f"{self.learning_algorithm.name}_{self.model_domain.name}_" \
                                                  f"{fold_num}_semantic_performance.csv"
        with open(statistics_path, "wt", newline='') as statistics_file:
            stats_writer = csv.DictWriter(statistics_file, fieldnames=SEMANTIC_PRECISION_STATS)
            stats_writer.writeheader()
            for data_line in self.combined_stats:
                stats_writer.writerow(data_line)

    def export_combined_semantic_performance(self) -> None:
        """Export the numeric learning statistics to a CSV report file."""
        statistics_path = self.results_dir_path / f"{self.learning_algorithm.name}_{self.model_domain.name}" \
                                                  "combined_semantic_performance.csv"
        with open(statistics_path, "wt", newline='') as statistics_file:
            stats_writer = csv.DictWriter(statistics_file, fieldnames=SEMANTIC_PRECISION_STATS)
            stats_writer.writeheader()
            for data_line in self.combined_stats:
                stats_writer.writerow(data_line)
