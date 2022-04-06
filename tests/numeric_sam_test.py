"""module tests for the Numeric SAM learning algorithm"""

from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser
from pddl_plus_parser.models import Domain, ActionCall, Problem, Observation, \
    ObservedComponent
from pytest import fixture

from sam_learning.learners import SAMLearner
from sam_learning.learners.numeric_sam import NumericSAMLearner
from tests.consts import NUMERIC_DOMAIN_PATH, \
    NUMERIC_PROBLEM_PATH, DEPOT_NUMERIC_TRAJECTORY_PATH


@fixture()
def depot_domain() -> Domain:
    domain_parser = DomainParser(NUMERIC_DOMAIN_PATH, partial_parsing=True)
    return domain_parser.parse_domain()


@fixture()
def depot_problem(depot_domain: Domain) -> Problem:
    return ProblemParser(problem_path=NUMERIC_PROBLEM_PATH, domain=depot_domain).parse_problem()


@fixture()
def numeric_observation(depot_domain: Domain, depot_problem: Problem) -> Observation:
    return TrajectoryParser(depot_domain, depot_problem).parse_trajectory(DEPOT_NUMERIC_TRAJECTORY_PATH)


@fixture()
def numeric_sam_learning(depot_domain: Domain) -> NumericSAMLearner:
    return NumericSAMLearner(depot_domain)


# def test_add_new_action_with_single_trajectory_component_adds_action_data_to_learned_domain(
#         sam_learning: SAMLearner, elevators_observation: Observation):
#     observation_component = elevators_observation.components[0]
#     previous_state = observation_component.previous_state
#     next_state = observation_component.next_state
#     test_action_call = observation_component.grounded_action_call
#
#     sam_learning.add_new_action(grounded_action=test_action_call, previous_state=previous_state, next_state=next_state)
#
#     added_action_name = "move-down-slow"
#
#     assert added_action_name in sam_learning.partial_domain.actions
#     learned_action_data = sam_learning.partial_domain.actions[added_action_name]
#     preconditions_str = set([p.untyped_representation for p in learned_action_data.positive_preconditions])
#     assert preconditions_str.issuperset(["(lift-at ?lift ?f1)", "(above ?f2 ?f1)", "(reachable-floor ?lift ?f2)"])
#     assert [p.untyped_representation for p in learned_action_data.add_effects] == ["(lift-at ?lift ?f2)"]
#     assert [p.untyped_representation for p in learned_action_data.delete_effects] == ["(lift-at ?lift ?f1)"]
#
#
# def test_update_action_with_two_trajectory_component_updates_action_data_correctly(
#         sam_learning: SAMLearner, elevators_observation: Observation):
#     first_observation_component = elevators_observation.components[0]
#     second_observation_component = elevators_observation.components[4]
#     first_action_call = ActionCall(name="move-down-slow", grounded_parameters=["slow2-0", "n17", "n16"])
#     second_action_call = ActionCall(name="move-down-slow", grounded_parameters=["slow1-0", "n9", "n8"])
#
#     sam_learning.add_new_action(grounded_action=first_action_call,
#                                 previous_state=first_observation_component.previous_state,
#                                 next_state=first_observation_component.next_state)
#
#     print(second_observation_component.previous_state.serialize())
#
#     sam_learning.update_action(grounded_action=second_action_call,
#                                previous_state=second_observation_component.previous_state,
#                                next_state=second_observation_component.next_state)
#     added_action_name = "move-down-slow"
#
#     assert added_action_name in sam_learning.partial_domain.actions
#     learned_action_data = sam_learning.partial_domain.actions[added_action_name]
#     preconditions_str = set([p.untyped_representation for p in learned_action_data.positive_preconditions])
#     assert preconditions_str.issuperset(["(lift-at ?lift ?f1)", "(above ?f2 ?f1)", "(reachable-floor ?lift ?f2)"])
#     assert [p.untyped_representation for p in learned_action_data.add_effects] == ["(lift-at ?lift ?f2)"]
#     assert [p.untyped_representation for p in learned_action_data.delete_effects] == ["(lift-at ?lift ?f1)"]
#
#
# def test_handle_single_trajectory_component_not_allowing_actions_with_duplicated_parameters(
#         sam_learning: SAMLearner, elevators_observation: Observation):
#     observation_component = elevators_observation.components[0]
#     test_action_call = ActionCall(name="move-down-slow", grounded_parameters=["slow2-0", "n17", "n17"])
#     component = ObservedComponent(observation_component.previous_state, test_action_call, observation_component.next_state)
#     sam_learning.handle_single_trajectory_component(component)
#
#     added_action_name = "move-down-slow"
#     learned_action_data = sam_learning.partial_domain.actions[added_action_name]
#     assert len(learned_action_data.positive_preconditions) == 0
#     assert len(learned_action_data.add_effects) == 0
#     assert len(learned_action_data.delete_effects) == 0
#
#
# def test_handle_single_trajectory_component_learns_preconditions_and_effects_when_given_a_non_duplicate_component(
#         sam_learning: SAMLearner, elevators_observation: Observation):
#     observation_component = elevators_observation.components[0]
#     sam_learning.handle_single_trajectory_component(observation_component)
#
#     added_action_name = "move-down-slow"
#     learned_action_data = sam_learning.partial_domain.actions[added_action_name]
#
#     preconditions_str = set([p.untyped_representation for p in learned_action_data.positive_preconditions])
#     assert preconditions_str.issuperset(["(lift-at ?lift ?f1)", "(above ?f2 ?f1)", "(reachable-floor ?lift ?f2)"])
#     assert [p.untyped_representation for p in learned_action_data.add_effects] == ["(lift-at ?lift ?f2)"]
#     assert [p.untyped_representation for p in learned_action_data.delete_effects] == ["(lift-at ?lift ?f1)"]

def test_learn_action_model_returns_learned_model(numeric_sam_learning: NumericSAMLearner, numeric_observation: Observation):
    learned_model = numeric_sam_learning.learn_action_model([numeric_observation])
    print()
    print(learned_model.to_pddl())