"""Module responsible for calculating our approach for numeric precision and recall."""
import logging
import random
from typing import List, Dict, NoReturn

from anytree import AnyNode
from pddl_plus_parser.models import Domain, PDDLFunction, evaluate_expression, Action


def set_expression_value(expression_node: AnyNode, grounded_fluents: Dict[str, float]) -> NoReturn:
    """Set the value of the expression according to the fluents present in the state.

    :param expression_node: the node that is currently being observed.
    :param grounded_fluents: the grounded numeric fluents present in the state.
    """
    if expression_node.is_leaf:
        if not isinstance(expression_node.value, PDDLFunction):
            return

        lifted_fluent: PDDLFunction = expression_node.value
        lifted_fluent.set_value(grounded_fluents[lifted_fluent.untyped_representation].value)
        return

    set_expression_value(expression_node.children[0], grounded_fluents)
    set_expression_value(expression_node.children[1], grounded_fluents)


NUM_SAMPLES = 1000


class NumericPerformanceCalculator:
    """Class responsible for calculating the numeric precision and recall of a model."""

    sample_points: List[float]
    model_domain: Domain
    precision_stats: List[Dict[str, float]]
    recall_stats: List[Dict[str, float]]
    mse_stats: List[Dict[str, float]]
    logger: logging.Logger

    def __init__(self, model_domain: Domain, relevant_fluents: Dict[str, List[str]]):
        self.logger = logging.getLogger(__name__)
        self.model_domain = model_domain
        self.relevant_fluents = relevant_fluents
        self.sample_points = [random.randint(0, 100) for _ in range(NUM_SAMPLES)]

    def calculate_performance(self, learned_domain: Domain, num_observations: int) -> NoReturn:
        """

        :param learned_domain:
        :param num_observations:
        :return:
        """
        iteration_stats = []
        for action_name, action_data in self.model_domain.actions.items():
            action_stats = {"num_trajectories": float(num_observations), "action_name": action_name}
            if action_name not in learned_domain.actions:
                action_stats["precision"] = 0
                action_stats["recall"] = 0

            elif len(self.relevant_fluents[action_name]) == 0:
                action_stats["precision"] = 1
                action_stats["recall"] = 1

            else:
                preconditions_performance = self.calculate_model_precondition_matches(
                    action_data, action_name, learned_domain)
                action_stats.update(preconditions_performance)

    def calculate_model_precondition_matches(self, action_data: Action, action_name: str,
                                             learned_domain: Domain) -> Dict[str, float]:
        num_true_positives = 0
        num_false_positives = 0
        num_false_negatives = 0
        for _ in range(NUM_SAMPLES):
            in_model_domain = True
            in_learned_domain = True
            grounded_fluents = {lifted_fluent_name: random.randint(0, 100) for lifted_fluent_name in
                                self.relevant_fluents[action_name]}
            for expression in action_data.numeric_preconditions:
                set_expression_value(expression.root, grounded_fluents)
                if not evaluate_expression(expression.root):
                    in_model_domain = False
                    break

            for expression in learned_domain.actions[action_name].numeric_preconditions:
                set_expression_value(expression.root, grounded_fluents)
                in_learned_domain = evaluate_expression(expression.root)
                if not in_learned_domain and not in_model_domain:
                    num_true_positives += 1
                    break
                elif not in_learned_domain and in_model_domain:
                    num_false_negatives += 1
                    break

            if in_model_domain and in_learned_domain:
                num_true_positives += 1
            else:
                num_false_positives += 1

        precision = num_true_positives / (num_true_positives + num_false_positives)
        recall = num_true_positives / (num_true_positives + num_false_negatives)
        return {"precision": precision, "recall": recall}
