"""Module test for Conditional SAM."""
from typing import Set

from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser
from pddl_plus_parser.models import Domain, Problem, Observation, GroundedPredicate, Predicate, ConditionalEffect, \
    PDDLType, State, ActionCall
from pytest import fixture

from sam_learning.core import DependencySet
from sam_learning.learners import ConditionalSAM
from sam_learning.learners.conditional_sam import extract_predicate_data
from tests.consts import SPIDER_DOMAIN_PATH, SPIDER_PROBLEM_PATH, SPIDER_TRAJECTORY_PATH, sync_snapshot


@fixture()
def spider_first_state(spider_observation: Observation) -> State:
    return spider_observation.components[0].previous_state


@fixture()
def spider_second_state(spider_observation: Observation) -> State:
    return spider_observation.components[0].next_state


@fixture()
def spider_first_action(spider_observation: Observation) -> ActionCall:
    return spider_observation.components[0].grounded_action_call


@fixture()
def spider_conditional_sam(spider_domain: Domain) -> ConditionalSAM:
    return ConditionalSAM(spider_domain, max_antecedents_size=1)


def test_initialize_actions_dependencies_adds_correct_dependencies(
        spider_conditional_sam: ConditionalSAM, spider_observation: Observation, spider_first_action: ActionCall):
    sync_snapshot(spider_conditional_sam, spider_observation.components[0], spider_observation.grounded_objects)
    spider_conditional_sam._initialize_actions_dependencies(spider_first_action)
    assert spider_conditional_sam.conditional_antecedents[spider_first_action.name] is not None


def test_update_observed_effects_adds_the_observed_effects_to_the_correct_set_for_conditional_effects(
        spider_conditional_sam: ConditionalSAM, spider_observation: Observation, spider_first_action: ActionCall,
        spider_first_state: State, spider_second_state: State):
    sync_snapshot(spider_conditional_sam, spider_observation.components[0], spider_observation.grounded_objects)

    spider_conditional_sam._initialize_actions_dependencies(spider_first_action)
    spider_conditional_sam._update_observed_effects(spider_first_action, spider_first_state, spider_second_state)
    assert len(spider_conditional_sam.observed_effects[spider_first_action.name]) > 0
    assert spider_conditional_sam.observed_effects[spider_first_action.name] == {"(currently-dealing )"}


def test_find_literals_not_in_state_correctly_sets_the_literals_that_do_not_appear_in_the_state(
        spider_conditional_sam: ConditionalSAM, spider_observation: Observation, spider_first_action: ActionCall):
    sync_snapshot(spider_conditional_sam, spider_observation.components[0], spider_observation.grounded_objects)

    predicates_not_in_state = spider_conditional_sam._find_literals_not_in_state(
        grounded_action=spider_first_action,
        grounded_predicates=spider_conditional_sam.triplet_snapshot.previous_state_positive_predicates.union(
            spider_conditional_sam.triplet_snapshot.previous_state_negative_predicates))

    negative_preconditions = {"(currently-updating-movable )", "(currently-updating-unmovable )",
                              "(currently-updating-part-of-tableau )", "(currently-collecting-deck )",
                              "(currently-dealing )"}

    assert negative_preconditions.issubset(predicates_not_in_state)


def test_find_literals_existing_in_state_correctly_sets_the_literals_that_do_appear_in_the_state(
        spider_conditional_sam: ConditionalSAM, spider_observation: Observation, spider_first_action: ActionCall):
    sync_snapshot(spider_conditional_sam, spider_observation.components[0], spider_observation.grounded_objects)

    predicates_not_in_state = spider_conditional_sam._find_literals_existing_in_state(
        grounded_action=spider_first_action,
        grounded_predicates=spider_conditional_sam.triplet_snapshot.previous_state_positive_predicates.union(
            spider_conditional_sam.triplet_snapshot.previous_state_negative_predicates))

    negative_preconditions_in_state = {"(not (currently-updating-movable ))", "(not (currently-updating-unmovable ))",
                                       "(not (currently-updating-part-of-tableau ))",
                                       "(not (currently-collecting-deck ))",
                                       "(not (currently-dealing ))"}

    assert negative_preconditions_in_state.issubset(predicates_not_in_state)


def test_remove_non_existing_previous_state_dependencies_removes_correct_predicates_from_literals_that_are_effects_only(
        spider_conditional_sam: ConditionalSAM, spider_observation: Observation, spider_first_action: ActionCall,
        spider_first_state: State, spider_second_state: State):
    sync_snapshot(spider_conditional_sam, spider_observation.components[0], spider_observation.grounded_objects)

    spider_conditional_sam._initialize_actions_dependencies(spider_first_action)
    spider_conditional_sam._update_observed_effects(spider_first_action, spider_first_state, spider_second_state)
    spider_conditional_sam._remove_non_existing_previous_state_dependencies(spider_first_action, spider_first_state,
                                                                            spider_second_state)
    not_dependencies = {"(currently-updating-movable )", "(currently-updating-unmovable )",
                        "(currently-updating-part-of-tableau )", "(currently-collecting-deck )",
                        "(currently-dealing )"}

    for not_dependency in not_dependencies:
        assert {not_dependency} not in \
               spider_conditional_sam.conditional_antecedents[spider_first_action.name].possible_antecedents[
                   "(currently-dealing )"]


def test_remove_existing_previous_state_dependencies_removes_literals_observed_in_previous_state_from_literals_that_cannot_be_effects(
        spider_conditional_sam: ConditionalSAM, spider_observation: Observation, spider_first_action: ActionCall,
        spider_first_state: State, spider_second_state: State):
    sync_snapshot(spider_conditional_sam, spider_observation.components[0], spider_observation.grounded_objects)

    spider_conditional_sam._initialize_actions_dependencies(spider_first_action)
    spider_conditional_sam._update_observed_effects(spider_first_action, spider_first_state, spider_second_state)
    spider_conditional_sam._remove_existing_previous_state_dependencies(spider_first_action)
    not_antecedents = {"(not (currently-updating-movable ))", "(not (currently-updating-unmovable ))",
                       "(not (currently-updating-part-of-tableau ))", "(not (currently-collecting-deck ))",
                       "(not (currently-dealing ))"}
    not_effect = "(not (currently-dealing ))"
    for not_antecedent in not_antecedents:
        assert {not_antecedent} not in \
               spider_conditional_sam.conditional_antecedents[spider_first_action.name].possible_antecedents[
                   not_effect]


def test_remove_existing_previous_state_dependencies_keeps_literals_not_observed_in_previous_state_in_literals_that_cannot_be_effects(
        spider_conditional_sam: ConditionalSAM, spider_observation: Observation, spider_first_action: ActionCall,
        spider_first_state: State, spider_second_state: State):
    sync_snapshot(spider_conditional_sam, spider_observation.components[0], spider_observation.grounded_objects)

    spider_conditional_sam._initialize_actions_dependencies(spider_first_action)
    spider_conditional_sam._update_observed_effects(spider_first_action, spider_first_state, spider_second_state)
    spider_conditional_sam._remove_existing_previous_state_dependencies(spider_first_action)
    possible_antecedents = {"(currently-updating-movable )", "(currently-updating-unmovable )",
                            "(currently-updating-part-of-tableau )", "(currently-collecting-deck )",
                            "(currently-dealing )"}
    not_effect = "(not (currently-dealing ))"
    for not_antecedent in possible_antecedents:
        assert {not_antecedent} in \
               spider_conditional_sam.conditional_antecedents[spider_first_action.name].possible_antecedents[
                   not_effect]


def test_update_observed_effects_adds_the_correct_literals_to_the_effect_with_the_correct_sign(
        spider_conditional_sam: ConditionalSAM, spider_observation: Observation, spider_first_action: ActionCall,
        spider_first_state: State, spider_second_state: State):
    sync_snapshot(spider_conditional_sam, spider_observation.components[0], spider_observation.grounded_objects)
    spider_conditional_sam._initialize_actions_dependencies(spider_first_action)
    spider_conditional_sam._update_observed_effects(spider_first_action, spider_first_state, spider_second_state)
    positive_effects = {"(currently-dealing )"}
    assert spider_conditional_sam.observed_effects[spider_first_action.name] == positive_effects


def test_apply_inductive_rules_updates_the_learned_effects_and_removes_not_antecedents(
        spider_conditional_sam: ConditionalSAM, spider_observation: Observation, spider_first_action: ActionCall,
        spider_first_state: State, spider_second_state: State):
    sync_snapshot(spider_conditional_sam, spider_observation.components[0], spider_observation.grounded_objects)
    spider_conditional_sam._initialize_actions_dependencies(spider_first_action)
    spider_conditional_sam._apply_inductive_rules(spider_first_action, spider_first_state, spider_second_state)

    positive_effect = "(currently-dealing )"
    not_effect = "(not (currently-dealing ))"

    assert spider_conditional_sam.observed_effects[spider_first_action.name] == {positive_effect}

    not_antecedents = {"(currently-updating-movable )", "(currently-updating-unmovable )",
                       "(currently-updating-part-of-tableau )", "(currently-collecting-deck )",
                       "(currently-dealing )"}
    for not_antecedent in not_antecedents:
        assert {not_antecedent} in \
               spider_conditional_sam.conditional_antecedents[spider_first_action.name].possible_antecedents[
                   not_effect]
        assert {not_antecedent} not in \
               spider_conditional_sam.conditional_antecedents[spider_first_action.name].possible_antecedents[
                   positive_effect]


def test_update_effects_data_updates_the_relevant_effects_and_removes_irrelevant_literals_from_dependency_set(
        spider_conditional_sam: ConditionalSAM, spider_observation: Observation, spider_first_action: ActionCall,
        spider_first_state: State, spider_second_state: State):
    sync_snapshot(spider_conditional_sam, spider_observation.components[0], spider_observation.grounded_objects)
    spider_conditional_sam._initialize_actions_dependencies(spider_first_action)
    initialized_add_effects = spider_conditional_sam.partial_domain.actions[spider_first_action.name].add_effects
    initialized_delete_effects = spider_conditional_sam.partial_domain.actions[spider_first_action.name].delete_effects
    spider_conditional_sam._apply_inductive_rules(spider_first_action, spider_first_state, spider_second_state)
    positive_dependencies = {"(currently-updating-movable )", "(currently-updating-unmovable )",
                             "(currently-updating-part-of-tableau )", "(currently-collecting-deck )",
                             "(currently-dealing )"}

    negative_dependencies = {f"(not {dep})" for dep in
                             positive_dependencies.difference({"(currently-collecting-deck )"})}

    tested_literal = "(not (currently-collecting-deck ))"
    assert len(spider_conditional_sam.partial_domain.actions[spider_first_action.name].add_effects) <= len(
        initialized_add_effects)
    assert len(spider_conditional_sam.partial_domain.actions[spider_first_action.name].delete_effects) <= len(
        initialized_delete_effects)

    for dependency in positive_dependencies:
        assert {dependency} in \
               spider_conditional_sam.conditional_antecedents[spider_first_action.name].possible_antecedents[
                   tested_literal]
        assert {dependency} not in \
               spider_conditional_sam.conditional_antecedents[spider_first_action.name].possible_antecedents[
                   "(currently-dealing )"]

    for dependency in negative_dependencies:
        assert {dependency} in \
               spider_conditional_sam.conditional_antecedents[grounded_action.name].possible_antecedents[
                   tested_literal]


def test_add_new_action_updates_action_negative_preconditions(spider_conditional_sam: ConditionalSAM,
                                                              spider_observation: Observation,
                                                              spider_domain: Domain):
    grounded_action = spider_observation.components[0].grounded_action_call
    spider_conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action,
        previous_state=spider_observation.components[0].previous_state,
        next_state=spider_observation.components[0].next_state)

    spider_conditional_sam._initialize_actions_dependencies(grounded_action)
    spider_conditional_sam.add_new_action(grounded_action,
                                          spider_observation.components[0].previous_state,
                                          spider_observation.components[0].next_state)

    added_action = spider_conditional_sam.partial_domain.actions[grounded_action.name]
    negative_preconditions = {f"(not {precondition.untyped_representation})" for precondition in
                              added_action.negative_preconditions}
    assert negative_preconditions.issuperset({"(not (currently-updating-movable ))",
                                              "(not (currently-updating-unmovable ))",
                                              "(not (currently-updating-part-of-tableau ))",
                                              "(not (currently-collecting-deck ))",
                                              "(not (currently-dealing ))"})


def test_add_new_action_updates_action_positive_preconditions(spider_conditional_sam: ConditionalSAM,
                                                              spider_observation: Observation,
                                                              spider_domain: Domain):
    grounded_action = spider_observation.components[0].grounded_action_call
    spider_conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action,
        previous_state=spider_observation.components[0].previous_state,
        next_state=spider_observation.components[0].next_state)

    spider_conditional_sam._initialize_actions_dependencies(grounded_action)
    spider_conditional_sam.add_new_action(grounded_action,
                                          spider_observation.components[0].previous_state,
                                          spider_observation.components[0].next_state)

    added_action = spider_conditional_sam.partial_domain.actions[grounded_action.name]
    positive_preconditions = {precondition.untyped_representation for precondition in
                              added_action.positive_preconditions}
    assert len(positive_preconditions) == 0


def test_add_new_action_updates_action_effects(spider_conditional_sam: ConditionalSAM,
                                               spider_observation: Observation,
                                               spider_domain: Domain):
    grounded_action = spider_observation.components[0].grounded_action_call
    spider_conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action,
        previous_state=spider_observation.components[0].previous_state,
        next_state=spider_observation.components[0].next_state)

    spider_conditional_sam._initialize_actions_dependencies(grounded_action)
    initialized_add_effects = spider_conditional_sam.partial_domain.actions[grounded_action.name].add_effects
    initialized_delete_effects = spider_conditional_sam.partial_domain.actions[grounded_action.name].delete_effects
    spider_conditional_sam.add_new_action(grounded_action,
                                          spider_observation.components[0].previous_state,
                                          spider_observation.components[0].next_state)

    added_action = spider_conditional_sam.partial_domain.actions[grounded_action.name]
    assert len(added_action.add_effects) <= len(initialized_add_effects)
    assert len(added_action.delete_effects) <= len(initialized_delete_effects)


def test_update_action_updates_preconditions(spider_conditional_sam: ConditionalSAM,
                                             spider_observation: Observation,
                                             spider_domain: Domain):
    grounded_action = spider_observation.components[0].grounded_action_call
    spider_conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action,
        previous_state=spider_observation.components[0].previous_state,
        next_state=spider_observation.components[0].next_state)

    spider_conditional_sam._initialize_actions_dependencies(grounded_action)
    spider_conditional_sam.add_new_action(grounded_action,
                                          spider_observation.components[0].previous_state,
                                          spider_observation.components[0].next_state)
    spider_conditional_sam.update_action(grounded_action,
                                         spider_observation.components[0].previous_state,
                                         spider_observation.components[0].next_state)

    added_action = spider_conditional_sam.partial_domain.actions[grounded_action.name]
    negative_preconditions = {f"(not {precondition.untyped_representation})" for precondition in
                              added_action.negative_preconditions}
    assert negative_preconditions.issuperset({"(not (currently-updating-movable ))",
                                              "(not (currently-updating-unmovable ))",
                                              "(not (currently-updating-part-of-tableau ))",
                                              "(not (currently-collecting-deck ))",
                                              "(not (currently-dealing ))"})


def test_extract_predicate_data_returns_correct_predicate_when_predicate_contains_no_parameters(
        spider_conditional_sam: ConditionalSAM, spider_domain: Domain):
    test_predicate = "(currently-updating-movable )"
    learner_action = spider_conditional_sam.partial_domain.actions["start-dealing"]
    result_predicate = extract_predicate_data(learner_action, test_predicate, spider_domain.constants)
    assert result_predicate.name == "currently-updating-movable"
    assert len(result_predicate.signature) == 0


def test_extract_predicate_data_returns_correct_predicate_predicate_contains_parameters(
        spider_conditional_sam: ConditionalSAM, spider_domain: Domain):
    test_predicate = "(to-deal ?c ?totableau ?fromdeal ?from)"
    learner_action = spider_conditional_sam.partial_domain.actions["deal-card"]
    result_predicate = extract_predicate_data(learner_action, test_predicate, spider_domain.constants)
    assert result_predicate.name == "to-deal"
    assert len(result_predicate.signature) == 4
    assert result_predicate.signature["?c"].name == "card"
    assert result_predicate.signature["?totableau"].name == "tableau"
    assert result_predicate.signature["?fromdeal"].name == "deal"
    assert result_predicate.signature["?from"].name == "cardposition"


def test_extract_predicate_data_returns_correct_predicate_predicate_contains_constants(
        spider_conditional_sam: ConditionalSAM, spider_domain: Domain):
    test_predicate = "(on ?c discard)"
    learner_action = spider_conditional_sam.partial_domain.actions["collect-card"]
    result_predicate = extract_predicate_data(learner_action, test_predicate, spider_domain.constants)
    assert result_predicate.name == "on"
    assert len(result_predicate.signature) == 2
    assert result_predicate.signature["?c"].name == "card"
    assert result_predicate.signature["discard"].name == "cardposition"


def test_extract_predicate_data_returns_correct_predicate_with_additional_type(
        spider_conditional_sam: ConditionalSAM, spider_domain: Domain):
    test_predicate = "(can-be-placed-on ?c ?c1)"
    learner_action = spider_conditional_sam.partial_domain.actions["collect-card"]
    result_predicate = extract_predicate_data(
        learner_action, test_predicate, spider_domain.constants,
        additional_parameter="?c1", additional_parameter_type=spider_domain.types["card"])
    assert result_predicate.name == "can-be-placed-on"
    assert len(result_predicate.signature) == 2
    assert result_predicate.signature["?c"].name == "card"
    assert result_predicate.signature["?c1"].name == "card"


def test_construct_conditional_effects_from_dependency_set_constructs_correct_conditional_effect(
        spider_conditional_sam: ConditionalSAM):
    dependecy_set = DependencySet(max_size_antecedents=1)
    dependecy_set.possible_antecedents = {
        "(currently-updating-unmovable )": [{"(not (can-continue-group ?c ?to))"}],
        "(make-unmovable ?to)": [{"(not (can-continue-group ?c ?to))"}]
    }
    test_action = spider_conditional_sam.partial_domain.actions["deal-card"]

    conditional_effect = spider_conditional_sam._construct_conditional_effect_data(test_action, dependecy_set,
                                                                                   "(make-unmovable ?to)")
    assert len(conditional_effect.negative_conditions) == 1
    assert conditional_effect.negative_conditions.pop().untyped_representation == "(can-continue-group ?c ?to)"
    assert len(conditional_effect.positive_conditions) == 0
    add_effect = conditional_effect.add_effects.pop().untyped_representation
    assert add_effect == "(make-unmovable ?to)"


def test_construct_restrictive_preconditions_constructs_correct_restrictive_precondition_string_as_required(
        spider_conditional_sam: ConditionalSAM):
    dependecy_set = DependencySet(max_size_antecedents=1)
    dependecy_set.possible_antecedents = {
        "(currently-updating-unmovable )": [{"(not (can-continue-group ?c ?to))"}],
        "(make-unmovable ?to)": [{"(not (can-continue-group ?c ?to))"}]
    }
    test_action = spider_conditional_sam.partial_domain.actions["deal-card"]

    spider_conditional_sam._construct_restrictive_preconditions(test_action, dependecy_set, "(make-unmovable ?to)")
    print(test_action.manual_preconditions)
    assert test_action.manual_preconditions == ["(or (make-unmovable ?to) (and (can-continue-group ?c ?to)))"]


def test_construct_restrictive_preconditions_constructs_correct_restrictive_precondition_string_as_required_when_is_effect(
        spider_conditional_sam: ConditionalSAM):
    dependecy_set = DependencySet(max_size_antecedents=1)
    dependecy_set.possible_antecedents = {
        "(currently-updating-unmovable )": [{"(not (can-continue-group ?c ?to))"}],
        "(make-unmovable ?to)": [{"(not (can-continue-group ?c ?to))"}]
    }
    test_action = spider_conditional_sam.partial_domain.actions["deal-card"]

    spider_conditional_sam.observed_effects[test_action.name].add("(make-unmovable ?to)")
    spider_conditional_sam._construct_restrictive_preconditions(test_action, dependecy_set, "(make-unmovable ?to)")
    print(test_action.manual_preconditions)
    assert test_action.manual_preconditions == [
        "(or (make-unmovable ?to) (and (can-continue-group ?c ?to)) (and (not (can-continue-group ?c ?to))))"]


def test_construct_restrictive_conditional_effects_constructs_the_correct_conditional_effect_in_the_action(
        spider_conditional_sam: ConditionalSAM):
    dependecy_set = DependencySet(max_size_antecedents=1)
    dependecy_set.possible_antecedents = {
        "(currently-updating-unmovable )": [{"(not (can-continue-group ?c ?to))"}],
        "(make-unmovable ?to)": [{"(not (can-continue-group ?c ?to))"}]
    }
    test_action = spider_conditional_sam.partial_domain.actions["deal-card"]

    spider_conditional_sam.observed_effects[test_action.name].add("(make-unmovable ?to)")
    conditional_effect = spider_conditional_sam._construct_restrictive_effect(test_action, dependecy_set,
                                                                              "(make-unmovable ?to)")
    assert str(conditional_effect) == "(when (and (not (can-continue-group ?c ?to))) (and (make-unmovable ?to)))"


def test_remove_preconditions_from_effects_removes_action_preconditions_from_dependencies(
        spider_conditional_sam: ConditionalSAM, spider_observation: Observation):
    spider_conditional_sam.current_trajectory_objects = spider_observation.grounded_objects
    spider_conditional_sam.handle_single_trajectory_component(spider_observation.components[0])
    spider_conditional_sam._remove_preconditions_from_antecedents(
        spider_conditional_sam.partial_domain.actions["start-dealing"])
    assert "(not (currently-updating-movable ))" not in spider_conditional_sam.conditional_antecedents[
        "start-dealing"].possible_antecedents
    assert "(not (currently-updating-unmovable ))" not in spider_conditional_sam.conditional_antecedents[
        "start-dealing"].possible_antecedents
    assert "(not (currently-updating-part-of-tableau ))" not in spider_conditional_sam.conditional_antecedents[
        "start-dealing"].possible_antecedents
    assert "(not (currently-collecting-deck ))" not in spider_conditional_sam.conditional_antecedents[
        "start-dealing"].possible_antecedents
    assert "(not (currently-dealing ))" not in spider_conditional_sam.conditional_antecedents[
        "start-dealing"].possible_antecedents


def test_handle_single_trajectory_component_learns_correct_information(
        spider_conditional_sam: ConditionalSAM, spider_observation: Observation):
    spider_conditional_sam.current_trajectory_objects = spider_observation.grounded_objects
    spider_conditional_sam.handle_single_trajectory_component(spider_observation.components[0])
    spider_conditional_sam._remove_preconditions_from_antecedents(
        spider_conditional_sam.partial_domain.actions["start-dealing"])
    pddl_action = spider_conditional_sam.partial_domain.actions["start-dealing"].to_pddl()
    assert "(not (currently-updating-unmovable ))" in pddl_action
    assert "(not (currently-updating-movable ))" in pddl_action
    assert "(not (currently-collecting-deck ))" in pddl_action
    assert "(not (currently-updating-part-of-tableau ))" in pddl_action
    assert "(not (currently-dealing ))" in pddl_action


def test_verify_and_construct_safe_conditional_effects_does_not_change_observed_effects(
        spider_conditional_sam: ConditionalSAM, spider_observation: Observation):
    first_component = spider_observation.components[0]
    spider_conditional_sam.current_trajectory_objects = spider_observation.grounded_objects
    spider_conditional_sam.handle_single_trajectory_component(first_component)
    test_action = spider_conditional_sam.partial_domain.actions[first_component.grounded_action_call.name]
    spider_conditional_sam._verify_and_construct_safe_conditional_effects(test_action)

    expected_effects = {"(robot-pos ?to)", "(not (robot-pos ?from))"}
    actual_effects = spider_conditional_sam.observed_effects[first_component.grounded_action_call.name]


def test_verify_and_construct_safe_conditional_effects_constructs_correct_safe_conditional_effects_with_observed_effects_only(
        spider_conditional_sam: ConditionalSAM, spider_observation: Observation):
    first_component = spider_observation.components[0]
    spider_conditional_sam.current_trajectory_objects = spider_observation.grounded_objects
    spider_conditional_sam.handle_single_trajectory_component(first_component)
    test_action = spider_conditional_sam.partial_domain.actions[first_component.grounded_action_call.name]
    spider_conditional_sam._remove_preconditions_from_antecedents(test_action)
    spider_conditional_sam._verify_and_construct_safe_conditional_effects(test_action)

    assert [effect.untyped_representation for effect in test_action.add_effects] == ["(currently-dealing )"]
    assert len(test_action.conditional_effects) == 0


def test_verify_and_construct_safe_conditional_effects_creates_conditional_effect_when_action_is_unsafe(
        spider_conditional_sam: ConditionalSAM, spider_observation: Observation):
    fourth_component = spider_observation.components[4]
    spider_conditional_sam.current_trajectory_objects = spider_observation.grounded_objects
    spider_conditional_sam.handle_single_trajectory_component(fourth_component)
    test_action = spider_conditional_sam.partial_domain.actions[fourth_component.grounded_action_call.name]
    spider_conditional_sam._verify_and_construct_safe_conditional_effects(test_action)

    assert len(test_action.conditional_effects) > 0


def test_compress_conditional_effects_compresses_conditional_effects_when_given_two_conditional_effects_with_the_same_antecedents(
        spider_conditional_sam: ConditionalSAM):
    antecedents = [Predicate("ante1", {}), Predicate("ante2", {})]
    results = [Predicate("result1", {}), Predicate("result2", {})]
    first_conditional_effect = ConditionalEffect()
    first_conditional_effect.positive_conditions = antecedents
    first_conditional_effect.add_effects.add(results[0])

    second_conditional_effect = ConditionalEffect()
    second_conditional_effect.positive_conditions = antecedents
    second_conditional_effect.add_effects.add(results[1])
    conditional_effects_list = [first_conditional_effect, second_conditional_effect]
    compressed_effects = spider_conditional_sam._compress_conditional_effects(conditional_effects_list)

    assert len(compressed_effects) == 1
    assert compressed_effects[0].positive_conditions == antecedents
    assert compressed_effects[0].add_effects == set(results)


def test_compress_conditional_effects_does_not_compress_effects_if_type_is_different_in_signature(
        spider_conditional_sam: ConditionalSAM):
    object_type = PDDLType("object")
    antecedents1 = [Predicate("ante1", {"?x": object_type}), Predicate("ante2", {"?x": object_type})]
    antecedent2 = [Predicate("ante1", {"?y": object_type}), Predicate("ante2", {"?x": object_type})]
    results = [Predicate("result1", {}), Predicate("result2", {})]
    first_conditional_effect = ConditionalEffect()
    first_conditional_effect.positive_conditions = antecedents1
    first_conditional_effect.add_effects.add(results[0])

    second_conditional_effect = ConditionalEffect()
    second_conditional_effect.positive_conditions = antecedent2
    second_conditional_effect.add_effects.add(results[1])
    conditional_effects_list = [first_conditional_effect, second_conditional_effect]
    compressed_effects = spider_conditional_sam._compress_conditional_effects(conditional_effects_list)

    assert len(compressed_effects) == 2


def test_compress_conditional_effects_does_compress_effects_even_when_unordered(spider_conditional_sam: ConditionalSAM):
    object_type = PDDLType("object")
    antecedents1 = [Predicate("ante1", {"?y": object_type}), Predicate("ante2", {"?x": object_type})]
    antecedent2 = [Predicate("ante2", {"?x": object_type}), Predicate("ante1", {"?y": object_type})]
    results = [Predicate("result1", {}), Predicate("result2", {})]
    first_conditional_effect = ConditionalEffect()
    first_conditional_effect.positive_conditions = antecedents1
    first_conditional_effect.add_effects.add(results[0])

    second_conditional_effect = ConditionalEffect()
    second_conditional_effect.positive_conditions = antecedent2
    second_conditional_effect.add_effects.add(results[1])
    conditional_effects_list = [first_conditional_effect, second_conditional_effect]
    compressed_effects = spider_conditional_sam._compress_conditional_effects(conditional_effects_list)

    assert len(compressed_effects) == 1


def test_compress_conditional_effects_compress_more_than_two_conditional_effects(
        spider_conditional_sam: ConditionalSAM):
    object_type = PDDLType("object")
    antecedents1 = [Predicate("ante1", {"?y": object_type}), Predicate("ante2", {"?x": object_type})]
    antecedent2 = [Predicate("ante2", {"?x": object_type}), Predicate("ante1", {"?y": object_type})]
    antecedent3 = [Predicate("ante2", {"?x": object_type}), Predicate("ante1", {"?y": object_type})]
    results = [Predicate("result1", {}), Predicate("result2", {}), Predicate("result3", {})]
    first_conditional_effect = ConditionalEffect()
    first_conditional_effect.positive_conditions = antecedents1
    first_conditional_effect.add_effects.add(results[0])

    second_conditional_effect = ConditionalEffect()
    second_conditional_effect.positive_conditions = antecedent2
    second_conditional_effect.add_effects.add(results[1])

    third_conditional_effect = ConditionalEffect()
    third_conditional_effect.positive_conditions = antecedent3
    third_conditional_effect.add_effects.add(results[2])

    conditional_effects_list = [first_conditional_effect, second_conditional_effect, third_conditional_effect]
    compressed_effects = spider_conditional_sam._compress_conditional_effects(conditional_effects_list)

    assert len(compressed_effects) == 1


def test_learn_action_model_learns_restrictive_action_mode(
        spider_conditional_sam: ConditionalSAM, spider_observation: Observation):
    learned_model, _ = spider_conditional_sam.learn_action_model([spider_observation])
    print(learned_model.to_pddl())
