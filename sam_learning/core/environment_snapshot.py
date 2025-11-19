"""Represents the current snapshot of the environment."""

import logging
from typing import Set, Dict, List

from pddl_plus_parser.models import GroundedPredicate, State, PDDLObject, Domain, ActionCall, PDDLFunction

from sam_learning.core.vocabulary_creator import VocabularyCreator


class EnvironmentSnapshot:
    """class representing the snapshot of the environment."""

    vocabulary_creator: VocabularyCreator
    previous_state_predicates: Set[GroundedPredicate]
    next_state_predicates: Set[GroundedPredicate]
    previous_state_functions: Dict[str, PDDLFunction]
    next_state_functions: Dict[str, PDDLFunction]
    partial_domain: Domain
    logger: logging.Logger

    def __init__(self, partial_domain: Domain):
        self.logger = logging.getLogger(__name__)
        self.vocabulary_creator = VocabularyCreator()
        self.previous_state_predicates = set()
        self.next_state_predicates = set()
        self.previous_state_functions = {}
        self.next_state_functions = {}
        self.partial_domain = partial_domain

    def create_discrete_state_snapshot(self, state: State, relevant_objects: List[PDDLObject]) -> Set[GroundedPredicate]:
        """Creates a snapshot of the state predicates.

        :param state: the state to create a snapshot of.
        :param relevant_objects: the relevant objects in the observation.
        """
        self.logger.debug("Creating a snapshot of the state predicates.")
        positive_state_predicates, negative_state_predicates = set(), set()
        vocabulary = self.vocabulary_creator.create_grounded_predicate_vocabulary(
            domain=self.partial_domain, observed_objects=relevant_objects
        )

        for lifted_predicate_name, vocabulary_predicates in vocabulary.items():
            # updates all the grounded predicates where there of a lifted predicate that does not appear in the state.
            if lifted_predicate_name not in state.state_predicates:
                negative_state_predicates.update([grounded_predicate.copy(is_negated=True) for grounded_predicate in vocabulary_predicates])
                continue

            for grounded_vocabulary_predicate in vocabulary_predicates:
                for predicate in state.state_predicates[lifted_predicate_name]:
                    if predicate.object_mapping == grounded_vocabulary_predicate.object_mapping:
                        positive_state_predicates.add(grounded_vocabulary_predicate.copy(is_negated=False))
                        break

            # Updates the grounded predicates of the predicates that appear in the state.
            negative_state_predicates.update(
                [
                    grounded_predicate.copy(is_negated=True)
                    for grounded_predicate in vocabulary_predicates.difference(positive_state_predicates)
                ]
            )

        return positive_state_predicates.union(negative_state_predicates)

    def create_numeric_state_snapshot(self, state: State, relevant_objects: List[PDDLObject]) -> Dict[str, PDDLFunction]:
        """Creating a snapshot of the state functions.

        :param state: the state to create a snapshot of.
        :param relevant_objects: the relevant objects of the action or all the possible objects in the observation.
        :return: the snapshot of the state functions.
        """
        self.logger.debug("Creating a snapshot of the state functions.")
        numeric_state_snapshot = {}
        object_names = {obj.name for obj in relevant_objects}
        for function_str, pddl_function in state.state_fluents.items():
            if len(pddl_function.signature.keys()) == 0 or set(pddl_function.signature.keys()).issubset(object_names):
                numeric_state_snapshot[function_str] = pddl_function

        return numeric_state_snapshot

    def create_triplet_snapshot(
        self,
        previous_state: State,
        next_state: State,
        current_action: ActionCall,
    ) -> None:
        """Creates a snapshot of the environment.

        :param previous_state: the previous state of the environment.
        :param next_state: the next state of the environment.
        :param current_action: the current action.
        """
        self.logger.debug("Creating a snapshot of the environment.")
        action_lifted_signature_types = list(self.partial_domain.actions[current_action.name].signature.values())
        parameterized_objects = [
            PDDLObject(name=param_name, type=param_type)
            for param_name, param_type in zip(current_action.parameters, action_lifted_signature_types)
        ] + list(self.partial_domain.constants.values())
        self.previous_state_predicates = self.create_discrete_state_snapshot(previous_state, parameterized_objects)
        self.next_state_predicates = self.create_discrete_state_snapshot(next_state, parameterized_objects)
        self.previous_state_functions = self.create_numeric_state_snapshot(previous_state, parameterized_objects)
        self.next_state_functions = self.create_numeric_state_snapshot(next_state, parameterized_objects)

    def create_triplet_snapshot_for_universal_variables(
        self,
        previous_state: State,
        next_state: State,
        current_action: ActionCall,
        observation_objects: Dict[str, PDDLObject],
        specific_types: List[str],
    ) -> None:
        """Creates a snapshot of the environment.

        :param previous_state: the previous state of the environment.
        :param next_state: the next state of the environment.
        :param current_action: the current action.
        :param observation_objects: the objects in the observation.
        :param specific_types: the types of the objects to include in the snapshot.
        """
        parameters_including_consts = current_action.parameters + list(self.partial_domain.constants.keys())
        relevant_objects = [
            object_data
            for object_data in {**observation_objects, **self.partial_domain.constants}.values()
            if object_data.name in parameters_including_consts or object_data.type.name in specific_types
        ]

        self.previous_state_predicates = self.create_discrete_state_snapshot(previous_state, relevant_objects)
        self.next_state_predicates = self.create_discrete_state_snapshot(next_state, relevant_objects)
        self.previous_state_functions = self.create_numeric_state_snapshot(previous_state, relevant_objects)
        self.next_state_functions = self.create_numeric_state_snapshot(next_state, relevant_objects)
