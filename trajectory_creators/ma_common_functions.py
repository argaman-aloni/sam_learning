"""Common functions for the multi-agent trajectory creators."""

import random
from pathlib import Path
from typing import List, Tuple

import numpy
from pddl_plus_parser.models import JointActionCall, NOP_ACTION, ActionCall, Problem, GroundedPredicate
from pddl_plus_parser.multi_agent import DUMMY_PREDICATE_NAME

RANDOM_PROBABILITIES = [0.35, 0.65]
ADD_PREDICATE_ACTION_NAME = "dummy-add-predicate-action"
DEL_PREDICATE_ACTION_NAME = "dummy-del-predicate-action"
DUMMY_PREDICATE = GroundedPredicate(name=DUMMY_PREDICATE_NAME, signature={}, object_mapping={}, is_positive=True)


def create_dataset_folders(workdir_path: Path) -> Tuple[Path, Path]:
    """Creates the folders for the dataset.

    :param workdir_path: the path to the working directory.
    :return: a tuple containing the paths to the enhanced and vanilla problems directories (with and without the dummy actions).
    """
    enhanced_problems_dir, vanilla_problems_dir = workdir_path / "enhanced", workdir_path / "vanilla"
    enhanced_problems_dir.mkdir(parents=True, exist_ok=True)
    vanilla_problems_dir.mkdir(parents=True, exist_ok=True)
    return enhanced_problems_dir, vanilla_problems_dir


def insert_dummy_predicate_to_goal(problem: Problem) -> bool:
    """Inserts the dummy predicate to the goal state.

    :param problem: the combined problem to insert the dummy predicate to.
    :return: whether the dummy predicate was inserted.
    """
    add_dummy_predicate = bool(random.randint(0, 1))
    if not add_dummy_predicate:
        return False

    problem.goal_state_predicates.append(DUMMY_PREDICATE)
    return True


def _find_empty_action_index(joint_action: JointActionCall) -> int:
    """Finds an empty action index in the joint action.

    :param joint_action: the joint action to find the empty action index in.
    :return: the index of the empty action.
    """
    empty_indices = [i for i, action in enumerate(joint_action.actions) if action.name == NOP_ACTION]
    if not empty_indices:
        # if all actions are filled, we need to find a new index
        return -1

    return random.choice(empty_indices)
def insert_dummy_actions_to_plan(
    plan_sequence: List[JointActionCall],
    agent_names: List[str],
    probabilities: List[float] = RANDOM_PROBABILITIES,
    dummy_in_goal: bool = False,
) -> List[JointActionCall]:
    """Inserts the dummy action to the plan sequence.

    :param plan_sequence: the joint action sequence to insert the dummy action to.
    :param agent_names: the names of the agents interacting with the environment.
    :param probabilities: the probabilities of inserting the dummy action.
    :param dummy_in_goal: whether the dummy predicate is in the goal and thus the dummy action should be called in the last plan sequence.
    :return: the joint action sequence with the dummy action inserted in multiple places.
    """
    new_plan_sequence = []
    last_empty_cell_to_add_dummy = len(plan_sequence) - 1
    if dummy_in_goal:
        # search for the last plan index with an empty slot to add the dummy action
        for index in range(len(plan_sequence) - 1, 0, -1):
            free_agent = _find_empty_action_index(plan_sequence[index])
            if free_agent != -1:
                last_empty_cell_to_add_dummy = index
                plan_sequence[index].actions[free_agent] = ActionCall(
                    name=ADD_PREDICATE_ACTION_NAME, grounded_parameters=[agent_names[free_agent]]
                )
                break

    for index, joint_action in enumerate(plan_sequence):
        if index >= last_empty_cell_to_add_dummy:
            # if we are past the last empty cell, we do not add any dummy actions
            new_plan_sequence.append(joint_action)
            continue

        add_dummy_action = bool(numpy.random.choice([0, 1], p=probabilities))
        should_be_delete_action = bool(random.randint(0, 1))
        if not add_dummy_action:
            continue

        new_action_index = _find_empty_action_index(joint_action)
        if new_action_index == -1:
            continue

        if should_be_delete_action:
            joint_action.actions[new_action_index] = ActionCall(
                name=DEL_PREDICATE_ACTION_NAME, grounded_parameters=[agent_names[new_action_index]]
            )
        else:
            joint_action.actions[new_action_index] = ActionCall(
                name=ADD_PREDICATE_ACTION_NAME, grounded_parameters=[agent_names[new_action_index]]
            )

        new_plan_sequence.append(joint_action)

    return new_plan_sequence
