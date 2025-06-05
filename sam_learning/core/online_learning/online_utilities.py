"""Utilities for online learning algorithms."""

from pathlib import Path
from typing import Dict, List

from pddl_plus_parser.exporters.numeric_trajectory_exporter import parse_action_call
from pddl_plus_parser.models import Precondition, Domain, ActionCall

from sam_learning.core.online_learning.online_discrete_models_learner import OnlineDiscreteModelLearner
from sam_learning.core.online_learning.online_numeric_models_learner import OnlineNumericModelLearner


def construct_safe_action_model(
    partial_domain: Domain,
    discrete_models_learners: Dict[str, OnlineDiscreteModelLearner],
    numeric_models_learners: Dict[str, OnlineNumericModelLearner],
) -> Domain:
    """Constructs the safe action model for the domain.

    :return: the safe action model.
    """
    safe_domain = partial_domain.shallow_copy()
    for action_name, action in safe_domain.actions.items():
        preconditions = Precondition("and")
        safe_discrete_preconditions, safe_discrete_effects = discrete_models_learners[action_name].get_safe_model()
        for precondition in safe_discrete_preconditions.operands:
            preconditions.add_condition(precondition)

        action.discrete_effects = safe_discrete_effects
        safe_numeric_preconditions, safe_numeric_effects = numeric_models_learners[action_name].get_safe_model()
        for precondition in safe_numeric_preconditions.operands:
            preconditions.add_condition(precondition)

        action.numeric_effects = safe_numeric_effects
        action.preconditions.root = preconditions

    return safe_domain


def construct_optimistic_action_model(
    partial_domain: Domain,
    discrete_models_learners: Dict[str, OnlineDiscreteModelLearner],
    numeric_models_learners: Dict[str, OnlineNumericModelLearner],
) -> Domain:
    """Constructs the optimistic action model for the domain.

    :return: the safe action model.
    """
    optimistic_domain = partial_domain.shallow_copy()
    optimistic_domain.requirements.add(":disjunctive-preconditions")
    for action_name, action in optimistic_domain.actions.items():
        preconditions = Precondition("and")
        optimistic_discrete_preconditions, optimistic_discrete_effects = discrete_models_learners[action_name].get_optimistic_model()
        for precondition in optimistic_discrete_preconditions.operands:
            preconditions.add_condition(precondition)

        action.discrete_effects = optimistic_discrete_effects
        optimistic_numeric_preconditions, optimistic_numeric_effects = numeric_models_learners[action_name].get_optimistic_model()
        for precondition in optimistic_numeric_preconditions.operands:
            preconditions.add_condition(precondition)

        action.numeric_effects = optimistic_numeric_effects
        action.preconditions.root = preconditions

    return optimistic_domain


def export_learned_domain(workdir: Path, partial_domain: Domain, learned_domain: Domain, is_safe_model: bool = True) -> Path:
    """Exports the learned domain into a file so that it will be used to solve the test set problems.

    :param workdir: the working directory where the domain file will be saved.23
    :param partial_domain: the partial domain that was used to learn the action model.
    :param learned_domain: the domain that was learned by the action model learning algorithm.
    :param is_safe_model: a boolean indicating whether the learned domain is a safe model or an optimistic model.
    :return: The path to the exported domain file.
    """
    domain_file_name = partial_domain.name + f"_{'safe' if is_safe_model else 'optimistic'}_learned_domain.pddl"
    domain_path = workdir / domain_file_name
    with open(domain_path, "wt") as domain_file:
        domain_file.write(learned_domain.to_pddl())

    return domain_path


def create_plan_actions(plan_path: Path) -> List[ActionCall]:
    """Reads a plan file and parses each line into an ActionCall.

    :param plan_path: Path to the plan file containing action calls.
    :return: List of ActionCall objects parsed from the plan file.
    """
    with open(plan_path, "rt") as plan_file:
        plan_lines = plan_file.readlines()

    plan_actions = []
    for line in plan_lines:
        plan_actions.append(parse_action_call(line))

    return plan_actions
