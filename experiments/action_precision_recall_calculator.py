from collections import defaultdict
from typing import Dict, Set, NoReturn, List

from pddl_plus_parser.models import Action

from sam_learning.core import LearnerAction

PRECISION_RECALL_FIELD_NAMES = [
    "preconditions_precision",
    "add_effects_precision",
    "delete_effects_precision",
    "preconditions_recall",
    "add_effects_recall",
    "delete_effects_recall",
    "action_precision",
    "action_recall",
    "f1_score"
]


def calculate_true_positive_rate(learned_predicates: Set[str], expected_predicates: Set[str]) -> int:
    """Calculates the number of predicates that appear both in the model domain and in the learned domain.

    :param learned_predicates: the predicates that belong in the learned domain.
    :param expected_predicates: the predicates that belong to the model domain.
    :return: the number of predicates that match in both domains.
    """
    return len(learned_predicates.intersection(expected_predicates))


def calculate_false_positive_rate(learned_predicates: Set[str], expected_predicates: Set[str]) -> int:
    """Calculates the number of predicates that appear in the learned domain but are not in the model domain.

    :param learned_predicates: the predicates that belong in the learned domain.
    :param expected_predicates: the predicates that belong to the model domain.
    :return: the number of predicates that belong to the learned domain and not the model domain.
    """
    return len(learned_predicates.difference(expected_predicates))


def calculate_false_negative_rate(learned_predicates: Set[str], expected_predicates: Set[str]) -> int:
    """Calculates the number of predicates that are missing in the learned domain from the model domain.

    :param learned_predicates: the predicates that belong in the learned domain.
    :param expected_predicates: the predicates that belong to the model domain.
    :return: the number of predicates missing in the learned domain.
    """
    return len(expected_predicates.difference(learned_predicates))


def calculate_recall(learned_predicates: Set[str], actual_predicates: Set[str]) -> float:
    """Calculates the recall value of the input predicates.

    :param learned_predicates: the predicates learned using the learning algorithm.
    :param actual_predicates: the predicates belonging to the model domain.
    :return: the recall value.
    """
    if len(learned_predicates) == 0:
        return 1

    if len(actual_predicates) == 0:
        return 0

    true_positives = calculate_true_positive_rate(learned_predicates, actual_predicates)
    false_negatives = calculate_false_negative_rate(learned_predicates, actual_predicates)
    return true_positives / (true_positives + false_negatives)


def calculate_precision(learned_predicates: Set[str], actual_predicates: Set[str]) -> float:
    """Calculates the precision value of the input predicates.

    :param learned_predicates: the predicates learned using the learning algorithm.
    :param actual_predicates: the predicates belonging to the model domain.
    :return: the precision value.
    """
    if len(learned_predicates) == 0:
        if len(actual_predicates) == 0:
            return 1

        return 0

    true_positives = calculate_true_positive_rate(learned_predicates, actual_predicates)
    false_positives = calculate_false_positive_rate(learned_predicates, actual_predicates)
    return true_positives / (true_positives + false_positives)


class PrecisionRecallCalculator:
    preconditions: Dict[str, Set[str]]
    ground_truth_preconditions: Dict[str, Set[str]]
    add_effects: Dict[str, Set[str]]
    ground_truth_add_effects: Dict[str, Set[str]]
    delete_effects: Dict[str, Set[str]]
    ground_truth_delete_effects: Dict[str, Set[str]]
    _observed_action_names: List[str]
    _unobserved_action_names: List[str]

    def __init__(self):
        self.preconditions = defaultdict(set)
        self.ground_truth_preconditions = defaultdict(set)
        self.add_effects = defaultdict(set)
        self.ground_truth_add_effects = defaultdict(set)
        self.delete_effects = defaultdict(set)
        self.ground_truth_delete_effects = defaultdict(set)
        self._compared_tuples = [(self.preconditions, self.ground_truth_preconditions),
                                 (self.add_effects, self.ground_truth_add_effects),
                                 (self.delete_effects, self.ground_truth_delete_effects)]
        self._observed_action_names = []
        self._unobserved_action_names = []

    def add_action_data(self, learned_action: LearnerAction, model_action: Action) -> NoReturn:
        """

        :param learned_action:
        :param model_action:
        :return:
        """
        self._observed_action_names.append(learned_action.name)
        self.preconditions[learned_action.name] = \
            {p.untyped_representation for p in learned_action.positive_preconditions}
        self.ground_truth_preconditions[model_action.name] = \
            {p.untyped_representation for p in model_action.positive_preconditions}
        self.add_effects[learned_action.name] = {p.untyped_representation for p in learned_action.add_effects}
        self.ground_truth_add_effects[model_action.name] = {p.untyped_representation for p in model_action.add_effects}
        self.delete_effects[learned_action.name] = {p.untyped_representation for p in learned_action.delete_effects}
        self.ground_truth_delete_effects[model_action.name] = \
            {p.untyped_representation for p in model_action.delete_effects}

    def add_unobserved_action(self, action_name: str) -> NoReturn:
        """Adds an action that was not observed in the trajectory to the appropriate list.
        
        :param action_name: the name of the action that was not observed in the trajectory.
        """
        self._unobserved_action_names.append(action_name)

    def calculate_action_precision(self, action_name: str) -> float:
        """calculates the precision value of a certain action.

        :param action_name: the name of the action that is being tested.
        :return: the action's precision.
        """
        if action_name in self._unobserved_action_names:
            return 0

        true_positives = sum(
            calculate_true_positive_rate(tup[0][action_name], tup[1][action_name]) for tup in self._compared_tuples)
        false_positives = sum(
            calculate_false_positive_rate(tup[0][action_name], tup[1][action_name]) for tup in self._compared_tuples)
        if true_positives == 0 and false_positives == 0:
            return 1

        return true_positives / (true_positives + false_positives)

    def calculate_action_recall(self, action_name: str) -> float:
        """calculates the recall value of a certain action.

        :param action_name: the name of the action that is being tested.
        :return: the action's recall.
        """
        if action_name in self._unobserved_action_names:
            return 0

        true_positives = sum(
            calculate_true_positive_rate(tup[0][action_name], tup[1][action_name]) for tup in self._compared_tuples)
        false_negatives = sum(
            calculate_false_negative_rate(tup[0][action_name], tup[1][action_name]) for tup in self._compared_tuples)

        if true_positives == 0 and false_negatives == 0:
            return 1

        return true_positives / (true_positives + false_negatives)

    def export_action_statistics(self, action_name: str) -> Dict[str, float]:
        """

        :param action_name:
        :return:
        """
        if action_name in self._unobserved_action_names:
            return {field: 0 for field in PRECISION_RECALL_FIELD_NAMES}

        action_precision = self.calculate_action_precision(action_name)
        action_recall = self.calculate_action_recall(action_name)
        action_f1_score = 2 * (action_precision * action_recall) / (action_precision + action_recall)
        return {
            "preconditions_precision": calculate_precision(self.preconditions[action_name],
                                                           self.ground_truth_preconditions[action_name]),
            "add_effects_precision": calculate_precision(self.add_effects[action_name],
                                                         self.ground_truth_add_effects[action_name]),
            "delete_effects_precision": calculate_precision(self.delete_effects[action_name],
                                                            self.ground_truth_delete_effects[action_name]),
            "preconditions_recall": calculate_recall(self.preconditions[action_name],
                                                     self.ground_truth_preconditions[action_name]),
            "add_effects_recall": calculate_recall(self.add_effects[action_name],
                                                   self.ground_truth_add_effects[action_name]),
            "delete_effects_recall": calculate_recall(self.delete_effects[action_name],
                                                      self.ground_truth_delete_effects[action_name]),
            "action_precision": action_precision,
            "action_recall": action_recall,
            "f1_score": action_f1_score
        }

    def calculate_model_precision(self) -> float:
        """calculates the precision value of a learned domain.

        :return: the model's precision.
        """
        if len(self._observed_action_names) == 0:
            return 0

        true_positives = sum([sum(calculate_true_positive_rate(tup[0][action_name], tup[1][action_name]) for tup in
                                  self._compared_tuples) for action_name in self._observed_action_names])
        false_positives = sum([sum(calculate_false_positive_rate(tup[0][action_name], tup[1][action_name]) for tup in
                                   self._compared_tuples) for action_name in self._observed_action_names])
        if true_positives == 0 and false_positives == 0:
            return 1

        return true_positives / (true_positives + false_positives)

    def calculate_model_recall(self) -> float:
        """calculates the recall value of a learned domain.

        :return: the model's recall.
        """
        if len(self._observed_action_names) == 0:
            return 0

        true_positives = sum([sum(calculate_true_positive_rate(tup[0][action_name], tup[1][action_name]) for tup in
                                  self._compared_tuples) for action_name in self._observed_action_names])
        false_negatives = sum([sum(calculate_false_negative_rate(tup[0][action_name], tup[1][action_name]) for tup in
                                   self._compared_tuples) for action_name in self._observed_action_names])
        if true_positives == 0 and false_negatives == 0:
            return 1

        return true_positives / (true_positives + false_negatives)
