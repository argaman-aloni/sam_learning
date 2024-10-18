"""Module to collect domain statistics."""
import logging
from collections import defaultdict
from typing import Dict, List, Set

from pddl_plus_parser.models import Domain, Observation, ActionCall

from sam_learning.core import NumericFunctionMatcher, PredicatesMatcher, EnvironmentSnapshot


class DomainStatisticsCollector:

    action_pre_state_numeric_statistics: Dict[str, Dict[str, Set[float]]]
    action_post_state_numeric_statistics: Dict[str, Dict[str, Set[float]]]

    def __init__(self, partial_domain: Domain):
        self.logger = logging.getLogger(__name__)
        self.action_pre_state_numeric_statistics = {action_name: defaultdict(set) for action_name in partial_domain.actions}
        self.action_post_state_numeric_statistics = {action_name: defaultdict(set) for action_name in partial_domain.actions}
        self._action_appearances = {action_name: 0 for action_name in partial_domain.actions}
        self._function_matcher = NumericFunctionMatcher(partial_domain)
        self._matcher = PredicatesMatcher(partial_domain)
        self._triplet_snapshot = EnvironmentSnapshot(partial_domain=partial_domain)
        self._partial_domain = partial_domain

    def collect_statistics(self, observations: List[Observation], fold: int) -> None:
        """Collects the statistics from the observations."""
        self.logger.info("Collecting the statistics from the observations.")
        for observation in observations:
            self._collect_statistics_from_observation(observation)

        self.logger.info("Done collecting the statistics from the observations.")
        self.logger.info("Combining the statistics from the observations and calculating functions' unique appearances.")
        self._combine_statistics_and_calculate_unique_appearances(fold, len(observations))

    def _combine_statistics_and_calculate_unique_appearances(self, fold, num_trajectories: int) -> None:
        """

        :param fold:
        :return:
        """
        for action in self._partial_domain.actions:
            average_unique_pre_state_functions = sum(
                len(function_values) for function_values in self.action_pre_state_numeric_statistics[action].values()
            ) / len(self.action_pre_state_numeric_statistics[action])
            average_unique_post_state_functions = sum(
                len(function_values) for function_values in self.action_post_state_numeric_statistics[action].values()
            ) / len(self.action_post_state_numeric_statistics[action])
            max_unique_pre_state_functions = max(
                len(function_values) for function_values in self.action_pre_state_numeric_statistics[action].values()
            )
            max_unique_post_state_functions = max(
                len(function_values) for function_values in self.action_post_state_numeric_statistics[action].values()
            )
            min_unique_pre_state_functions = min(
                len(function_values) for function_values in self.action_pre_state_numeric_statistics[action].values()
            )
            min_unique_post_state_functions = min(
                len(function_values) for function_values in self.action_post_state_numeric_statistics[action].values()
            )

            combined_pre_state_statistics = {
                "num_trajectories": num_trajectories,
                "action_name": action,
                "num_appearances": self._action_appearances[action],
                **{function_name: len(function_values) for function_name, function_values in self.action_pre_state_propositional_statistics[action]},
                **self.action_pre_state_numeric_statistics[action],
            }
            combined_post_state_statistics = {
                "num_trajectories": num_trajectories,
                "action_name": action,
                "num_appearances": self._action_appearances[action],
                **{function_name: len(function_values) for function_name, function_values in self.action_post_state_propositional_statistics[action]},
                **self.action_post_state_numeric_statistics[action],
            }

        self._save_statistics_to_file(fold)

    def _collect_statistics_from_observation(self, observation: Observation) -> None:
        """Collects the statistics from the given observation.

        :param observation: the observation to collect the statistics from.
        """
        for component in observation.components:
            previous_state = component.previous_state
            grounded_action = component.grounded_action_call
            next_state = component.next_state
            self._action_appearances[grounded_action.name] += 1
            self._triplet_snapshot.create_triplet_snapshot(
                previous_state=previous_state, next_state=next_state, current_action=grounded_action, observation_objects=observation.grounded_objects
            )
            self._add_action_to_statistics(grounded_action)

    def _add_action_to_statistics(self, grounded_action: ActionCall) -> None:
        """

        :param grounded_action:
        :return:
        """
        previous_state_lifted_matches = self._function_matcher.match_state_functions(grounded_action, self._triplet_snapshot.previous_state_functions)
        next_state_lifted_matches = self._function_matcher.match_state_functions(grounded_action, self._triplet_snapshot.next_state_functions)
        for function in previous_state_lifted_matches.values():
            self.action_pre_state_numeric_statistics[grounded_action.name][function.untyped_representation].add(function.value)

        for function in next_state_lifted_matches.values():
            self.action_post_state_numeric_statistics[grounded_action.name][function.untyped_representation].add(function.value)
