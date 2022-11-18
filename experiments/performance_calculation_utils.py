"""Utilities for the performance calculation process."""

from typing import Dict

from pddl_plus_parser.models import Domain, Operator, ActionCall, State


def _ground_tested_operator(action_call: ActionCall, learned_domain: Domain) -> Operator:
    """Ground the tested action based on the trajectory data.

    :param action_call: the grounded action call in the observation component.
    :param learned_domain: the domain that was learned using the action model learning algorithm.
    :return: the grounded operator.
    """
    grounded_operator = Operator(
        action=learned_domain.actions[action_call.name],
        domain=learned_domain,
        grounded_action_call=action_call.parameters)
    grounded_operator.ground()
    return grounded_operator


def _calculate_single_action_applicability_rate(
        action_call: ActionCall, learned_domain: Domain, complete_domain: Domain,
        num_false_negatives: Dict[str, int], num_false_positives: Dict[str, int],
        num_true_positives: Dict[str, int], observed_state: State):
    """

    :param action_call:
    :param learned_domain:
    :param complete_domain:
    :param num_false_negatives:
    :param num_false_positives:
    :param num_true_positives:
    :param observed_state:
    :return:
    """
    tested_grounded_operator = _ground_tested_operator(action_call, learned_domain)
    model_grounded_operator = _ground_tested_operator(action_call, complete_domain)
    is_applicable_in_test = tested_grounded_operator.is_applicable(observed_state)
    is_applicable_in_model = model_grounded_operator.is_applicable(observed_state)
    num_true_positives[action_call.name] += int(is_applicable_in_test == is_applicable_in_model)
    num_false_positives[action_call.name] += int(is_applicable_in_test and not is_applicable_in_model)
    num_false_negatives[action_call.name] += int(not is_applicable_in_test and is_applicable_in_model)
