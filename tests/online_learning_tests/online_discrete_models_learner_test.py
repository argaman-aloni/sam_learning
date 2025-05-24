"""Module test for the propositional information gain functionality."""
from typing import Set

import pytest
from pddl_plus_parser.models import Domain, Predicate, Precondition

from sam_learning.core import OnlineDiscreteModelLearner, VocabularyCreator
from sam_learning.core.online_learning.online_discrete_models_learner import DUMMY_EFFECT

TEST_ACTION_NAME = "test_action"


@pytest.fixture
def lifted_vocabulary(woodworking_domain: Domain) -> Set[Predicate]:
    return VocabularyCreator().create_lifted_vocabulary(
        domain=woodworking_domain, possible_parameters=woodworking_domain.actions["do-grind"].signature
    )


@pytest.fixture
def depot_lifted_vocabulary(depot_domain: Domain) -> Set[Predicate]:
    return VocabularyCreator().create_lifted_vocabulary(
        domain=depot_domain, possible_parameters=depot_domain.actions["lift"].signature
    )


@pytest.fixture
def online_discrete_model_learner(lifted_vocabulary) -> OnlineDiscreteModelLearner:
    return OnlineDiscreteModelLearner(TEST_ACTION_NAME, lifted_vocabulary)


@pytest.fixture
def depot_online_discrete_model_learner(depot_lifted_vocabulary) -> OnlineDiscreteModelLearner:
    return OnlineDiscreteModelLearner("lift", depot_lifted_vocabulary)


def test_initialization_of_learner_with_empty_predicates_set_works_correctly():
    empty_predicates_set = set()
    online_discrete_model_learner = OnlineDiscreteModelLearner(TEST_ACTION_NAME, empty_predicates_set)
    assert online_discrete_model_learner.action_name == TEST_ACTION_NAME
    assert len(online_discrete_model_learner.predicates_superset) == 0
    assert len(online_discrete_model_learner.cannot_be_preconditions) == 0
    assert len(online_discrete_model_learner.must_be_preconditions) == 0
    assert len(online_discrete_model_learner.cannot_be_effects) == 0
    assert len(online_discrete_model_learner.must_be_effects) == 0


def test_add_positive_post_state_observation_when_no_preconditions_exists_adds_it_to_the_superset_creates_a_complementary_set_for_the_not_preconditions_and_returns_none(
        online_discrete_model_learner: OnlineDiscreteModelLearner, lifted_vocabulary: Set[Predicate]
):
    new_positive_sample = {p for p in lifted_vocabulary if
                           p.untyped_representation in ["(available ?x)", "(treatment ?x ?oldtreatment)"]}
    online_discrete_model_learner._add_positive_pre_state_observation(new_positive_sample)
    assert len(online_discrete_model_learner.cannot_be_preconditions) == len(lifted_vocabulary) - 2
    assert len(online_discrete_model_learner.must_be_preconditions) == 0


def test_add_positive_post_state_observation_when_sample_already_added_does_not_change_sets(
        online_discrete_model_learner: OnlineDiscreteModelLearner, lifted_vocabulary: Set[Predicate]
):
    new_positive_sample = {p for p in lifted_vocabulary if
                           p.untyped_representation in ["(available ?x)", "(treatment ?x ?oldtreatment)"]}
    online_discrete_model_learner._add_positive_pre_state_observation(new_positive_sample)
    assert len(online_discrete_model_learner.cannot_be_preconditions) == len(lifted_vocabulary) - 2
    assert len(online_discrete_model_learner.must_be_preconditions) == 0
    online_discrete_model_learner._add_positive_pre_state_observation(new_positive_sample)
    assert len(online_discrete_model_learner.cannot_be_preconditions) == len(lifted_vocabulary) - 2
    assert len(online_discrete_model_learner.must_be_preconditions) == 0


def test_apply_unit_propagation_when_unit_clause_exists_that_is_part_on_a_non_unit_clause_cnf_removes_no_unit_clause_from_must_be_preconditions(
        online_discrete_model_learner: OnlineDiscreteModelLearner, lifted_vocabulary: Set[Predicate]
):
    unit_clause = {p for p in lifted_vocabulary if p.untyped_representation in ["(available ?x)"]}
    non_unit_clause = {p for p in lifted_vocabulary if
                       p.untyped_representation in ["(treatment ?x ?oldtreatment)", "(available ?x)"]}
    online_discrete_model_learner.must_be_preconditions.append(non_unit_clause)
    online_discrete_model_learner.must_be_preconditions.append(unit_clause)
    assert len(online_discrete_model_learner.must_be_preconditions) == 2
    online_discrete_model_learner._apply_unit_propagation()
    assert len(online_discrete_model_learner.must_be_preconditions) == 1


def test_apply_unit_propagation_when_unit_clauses_exists_and_no_non_unit_clause_exist_does_not_change_must_be_preconditions(
        online_discrete_model_learner: OnlineDiscreteModelLearner, lifted_vocabulary: Set[Predicate]
):
    unit_clause = {p for p in lifted_vocabulary if p.untyped_representation in ["(available ?x)"]}
    non_unit_clause = {p for p in lifted_vocabulary if p.untyped_representation in ["(treatment ?x ?oldtreatment)"]}
    online_discrete_model_learner.must_be_preconditions.append(non_unit_clause)
    online_discrete_model_learner.must_be_preconditions.append(unit_clause)
    assert len(online_discrete_model_learner.must_be_preconditions) == 2
    online_discrete_model_learner._apply_unit_propagation()
    assert len(online_discrete_model_learner.must_be_preconditions) == 2


def test_add_negative_pre_state_observation_adds_correct_set_of_missing_preconditions_to_the_must_be_preconditions_set(
        online_discrete_model_learner: OnlineDiscreteModelLearner, lifted_vocabulary: Set[Predicate]
):
    new_negative_sample = {p for p in lifted_vocabulary if
                           p.untyped_representation in ["(not (available ?x))", "(not (treatment ?x ?oldtreatment))"]}
    online_discrete_model_learner._add_negative_pre_state_observation(new_negative_sample)
    assert len(online_discrete_model_learner.must_be_preconditions) == 1
    must_be_pre = online_discrete_model_learner.must_be_preconditions[0]
    assert len(must_be_pre) == len(lifted_vocabulary) - 2
    assert {p.untyped_representation for p in must_be_pre}.issuperset(
        {"(available ?x)", "(treatment ?x ?oldtreatment)"})


def test_add_negative_pre_state_observation_and__add_positive_post_state_observation_adds_correct_set_of_missing_preconditions_to_the_must_be_preconditions_set(
        online_discrete_model_learner: OnlineDiscreteModelLearner, lifted_vocabulary: Set[Predicate]
):
    new_positive_sample = {
        p
        for p in lifted_vocabulary
        if p.untyped_representation
           in [
               "(available ?x)",
               "(surface-condition ?x ?oldsurface)",
               "(is-smooth ?oldsurface)",
               "(colour ?x ?oldcolour)",
               "(treatment ?x ?oldtreatment)",
               "(grind-treatment-change ?m ?oldtreatment ?newtreatment)",
           ]
    }
    online_discrete_model_learner._add_positive_pre_state_observation(new_positive_sample)
    new_negative_sample = {
        p
        for p in lifted_vocabulary
        if p.untyped_representation
           in [
               "(not (available ?x))",
               "(surface-condition ?x ?oldsurface)",
               "(is-smooth ?oldsurface)",
               "(colour ?x ?oldcolour)",
               "(not (treatment ?x ?oldtreatment))",
               "(grind-treatment-change ?m ?oldtreatment ?newtreatment)",
           ]
    }
    online_discrete_model_learner._add_negative_pre_state_observation(new_negative_sample)
    assert len(online_discrete_model_learner.must_be_preconditions) == 1
    must_be_pre = online_discrete_model_learner.must_be_preconditions[0]
    assert len(must_be_pre) == 2
    assert {p.untyped_representation for p in must_be_pre} == {"(available ?x)", "(treatment ?x ?oldtreatment)"}


def test_add_negative_pre_state_observation_when_trying_to_add_same_state_twice_does_not_add_state_again(
        online_discrete_model_learner: OnlineDiscreteModelLearner, lifted_vocabulary: Set[Predicate]
):
    new_negative_sample = {
        p
        for p in lifted_vocabulary
        if p.untyped_representation
           in [
               "(not (available ?x))",
               "(surface-condition ?x ?oldsurface)",
               "(is-smooth ?oldsurface)",
               "(colour ?x ?oldcolour)",
               "(not (treatment ?x ?oldtreatment))",
               "(grind-treatment-change ?m ?oldtreatment ?newtreatment)",
           ]
    }
    online_discrete_model_learner._add_negative_pre_state_observation(new_negative_sample)
    assert len(online_discrete_model_learner.must_be_preconditions) == 1
    online_discrete_model_learner._add_negative_pre_state_observation(new_negative_sample)
    assert len(online_discrete_model_learner.must_be_preconditions) == 1


def test_add_negative_pre_state_observation_and_add_positive_pre_state_observation_removes_redundant_preconditions_from_must_be_preconditions(
        online_discrete_model_learner: OnlineDiscreteModelLearner, lifted_vocabulary: Set[Predicate]
):
    new_positive_sample = {
        p
        for p in lifted_vocabulary
        if p.untyped_representation
           in [
               "(available ?x)",
               "(surface-condition ?x ?oldsurface)",
               "(is-smooth ?oldsurface)",
               "(colour ?x ?oldcolour)",
               "(treatment ?x ?oldtreatment)",
               "(grind-treatment-change ?m ?oldtreatment ?newtreatment)",
               "(not (grind-treatment-change ?m ?newtreatment ?oldtreatment))",
           ]
    }
    online_discrete_model_learner._add_positive_pre_state_observation(new_positive_sample)
    assert len(online_discrete_model_learner.predicates_superset.difference(
        online_discrete_model_learner.cannot_be_preconditions)) == 7
    new_negative_sample = {
        p
        for p in lifted_vocabulary
        if p.untyped_representation
           in [
               "(not (available ?x))",
               "(surface-condition ?x ?oldsurface)",
               "(is-smooth ?oldsurface)",
               "(colour ?x ?oldcolour)",
               "(not (treatment ?x ?oldtreatment))",
               "(grind-treatment-change ?m ?oldtreatment ?newtreatment)",
           ]
    }
    online_discrete_model_learner._add_negative_pre_state_observation(new_negative_sample)
    assert len(online_discrete_model_learner.must_be_preconditions) == 1
    must_be_pre = online_discrete_model_learner.must_be_preconditions[0]
    assert len(must_be_pre) == 3
    new_positive_sample = {
        p
        for p in lifted_vocabulary
        if p.untyped_representation
           in [
               "(available ?x)",
               "(surface-condition ?x ?oldsurface)",
               "(is-smooth ?oldsurface)",
               "(colour ?x ?oldcolour)",
               "(treatment ?x ?oldtreatment)",
               "(grind-treatment-change ?m ?oldtreatment ?newtreatment)",
           ]
    }
    online_discrete_model_learner._add_positive_pre_state_observation(new_positive_sample)
    must_be_pre = online_discrete_model_learner.must_be_preconditions[0]
    assert len(must_be_pre) == 2


def test_add_positive_post_state_observation_when_adding_sample_for_the_first_time_updates_must_be_effects_and_cannot_be_effects_correctly(
        online_discrete_model_learner: OnlineDiscreteModelLearner, lifted_vocabulary: Set[Predicate]
):
    pre_state_predicates = {
        p for p in lifted_vocabulary if
        p.untyped_representation not in ["(available ?x)", "(not (surface-condition ?x ?oldsurface))", ]
    }
    next_state_predicates = {
        p for p in lifted_vocabulary if
        p.untyped_representation in ["(available ?x)", "(not (surface-condition ?x ?oldsurface))", ]
    }
    online_discrete_model_learner._add_positive_post_state_observation(
        pre_state_predicates=pre_state_predicates, post_state_predicates=next_state_predicates
    )
    assert len(online_discrete_model_learner.must_be_effects) == 2
    print(len(online_discrete_model_learner.predicates_superset))
    assert len(online_discrete_model_learner.cannot_be_effects) == 2


def test_add_positive_post_state_observation_when_adding_sample_for_the_second_time_adds_additional_predicates_to_the_cannot_be_effects_set(
        depot_online_discrete_model_learner: OnlineDiscreteModelLearner, depot_lifted_vocabulary: Set[Predicate]
):
    pre_state_predicates = {
        p for p in depot_lifted_vocabulary if
        p.untyped_representation not in ["(not (at ?x ?p))", "(not (available ?x))", "(not (at ?y ?p))",
                                         "(not (clear ?y))", "(not (at ?z ?p))"]
    }
    next_state_predicates = {
        p for p in depot_lifted_vocabulary if
        p.untyped_representation not in ["(at ?y ?p)", "(clear ?y)", "(not (lifting ?x ?y))", "(available ?x)",
                                         "(not (clear ?z))", "(not (at ?z ?p))"]
    }
    depot_online_discrete_model_learner._add_positive_post_state_observation(
        pre_state_predicates=pre_state_predicates, post_state_predicates=next_state_predicates
    )
    assert "(at ?z ?p)" not in [eff.untyped_representation for eff in
                                depot_online_discrete_model_learner.cannot_be_effects]

    pre_state_predicates = {
        p for p in depot_lifted_vocabulary if
        p.untyped_representation not in ["(not (at ?x ?p))", "(not (available ?x))", "(not (at ?y ?p))",
                                         "(not (clear ?y))", "(at ?z ?p)"]
    }
    next_state_predicates = {
        p for p in depot_lifted_vocabulary if
        p.untyped_representation not in ["(at ?y ?p)", "(clear ?y)", "(not (lifting ?x ?y))", "(available ?x)",
                                         "(not (clear ?z))", "(at ?z ?p)"]
    }
    depot_online_discrete_model_learner._add_positive_post_state_observation(
        pre_state_predicates=pre_state_predicates, post_state_predicates=next_state_predicates
    )
    assert "(at ?z ?p)" in [eff.untyped_representation for eff in depot_online_discrete_model_learner.cannot_be_effects]


def test_add_transition_data_when_transition_is_successful_adds_positive_pre_and_post_state_observations(
        online_discrete_model_learner: OnlineDiscreteModelLearner, lifted_vocabulary: Set[Predicate]
):
    pre_state_predicates = {
        p for p in lifted_vocabulary if
        p.untyped_representation not in ["(not (available ?x))", "(surface-condition ?x ?oldsurface)", ]
    }
    next_state_predicates = {
        p for p in lifted_vocabulary if
        p.untyped_representation not in ["(available ?x)", "(not (surface-condition ?x ?oldsurface))", ]
    }
    online_discrete_model_learner.add_transition_data(pre_state_predicates, next_state_predicates,
                                                      is_transition_successful=True)
    assert len(online_discrete_model_learner.must_be_effects) == 2
    assert len(online_discrete_model_learner.cannot_be_effects) == len(lifted_vocabulary) - 2
    assert len(online_discrete_model_learner.must_be_preconditions) == 0
    assert len(online_discrete_model_learner.cannot_be_preconditions) == 2


def test_add_transition_data_when_transition_is_not_successful_adds_negative_pre_and_does_not_change_effects_data(
        online_discrete_model_learner: OnlineDiscreteModelLearner, lifted_vocabulary: Set[Predicate]
):
    pre_state_predicates = {
        p for p in lifted_vocabulary if
        p.untyped_representation not in ["(not (available ?x))", "(surface-condition ?x ?oldsurface)", ]
    }
    next_state_predicates = {
        p for p in lifted_vocabulary if
        p.untyped_representation not in ["(available ?x)", "(not (surface-condition ?x ?oldsurface))", ]
    }
    online_discrete_model_learner.add_transition_data(pre_state_predicates, next_state_predicates,
                                                      is_transition_successful=False)
    assert len(online_discrete_model_learner.must_be_effects) == 0
    assert len(online_discrete_model_learner.cannot_be_effects) == 0
    assert len(online_discrete_model_learner.must_be_preconditions) == 1
    assert len(online_discrete_model_learner.cannot_be_preconditions) == 0


def test_get_safe_model_returns_correct_safe_model_even_if_no_observation_was_given(
        online_discrete_model_learner: OnlineDiscreteModelLearner, lifted_vocabulary: Set[Predicate]
):
    safe_precondition, safe_effects = online_discrete_model_learner.get_safe_model()
    assert safe_precondition is not None
    assert len(safe_precondition.operands) == len(lifted_vocabulary)
    assert len(safe_effects) == 0


def test_get_safe_model_returns_empty_preconditions_and_effects_when_learner_initialized_with_no_predicates():
    empty_predicates_set = set()
    online_discrete_model_learner = OnlineDiscreteModelLearner(TEST_ACTION_NAME, empty_predicates_set)
    safe_precondition, safe_effects = online_discrete_model_learner.get_safe_model()
    assert safe_precondition is not None
    assert len(safe_precondition.operands) == 0
    assert len(safe_effects) == 0


def test_get_safe_model_when_observed_a_single_positive_observation_returns_preconditions_and_effects_based_on_the_observation(
        online_discrete_model_learner: OnlineDiscreteModelLearner, lifted_vocabulary: Set[Predicate]
):
    pre_state_predicates = {
        p for p in lifted_vocabulary if
        p.untyped_representation not in ["(not (available ?x))", "(surface-condition ?x ?oldsurface)", ]
    }
    next_state_predicates = {
        p for p in lifted_vocabulary if
        p.untyped_representation not in ["(available ?x)", "(not (surface-condition ?x ?oldsurface))", ]
    }
    online_discrete_model_learner.add_transition_data(pre_state_predicates, next_state_predicates,
                                                      is_transition_successful=True)
    safe_precondition, safe_effects = online_discrete_model_learner.get_safe_model()
    assert safe_precondition is not None
    assert len(safe_precondition.operands) == len(lifted_vocabulary) - 2
    assert len(safe_effects) == 2


def test_get_safe_model_when_observed_a_single_negative_observation_returns_preconditions_and_no_effects(
        online_discrete_model_learner: OnlineDiscreteModelLearner, lifted_vocabulary: Set[Predicate]
):
    pre_state_predicates = {
        p for p in lifted_vocabulary if
        p.untyped_representation not in ["(not (available ?x))", "(surface-condition ?x ?oldsurface)", ]
    }
    next_state_predicates = {
        p for p in lifted_vocabulary if
        p.untyped_representation not in ["(available ?x)", "(not (surface-condition ?x ?oldsurface))", ]
    }
    online_discrete_model_learner.add_transition_data(pre_state_predicates, next_state_predicates,
                                                      is_transition_successful=False)
    safe_precondition, safe_effects = online_discrete_model_learner.get_safe_model()
    assert safe_precondition is not None
    assert len(safe_precondition.operands) == len(lifted_vocabulary)
    assert len(safe_effects) == 0


def test_get_optimistic_model_when_no_observation_was_given_returns_effects_that_are_only_positive_predicates(
        online_discrete_model_learner: OnlineDiscreteModelLearner, lifted_vocabulary: Set[Predicate]
):
    optimistic_precondition, optimistic_effects = online_discrete_model_learner.get_optimistic_model()
    assert optimistic_precondition is not None
    assert len(optimistic_precondition.operands) == 0
    assert len(optimistic_effects) > 0
    assert all([eff.is_positive for eff in optimistic_effects])


def test_get_optimistic_model_returns_empty_preconditions_and_and_no_predicates_as_effects_when_learner_initialized_with_no_predicates():
    empty_predicates_set = set()
    online_discrete_model_learner = OnlineDiscreteModelLearner(TEST_ACTION_NAME, empty_predicates_set)
    safe_precondition, safe_effects = online_discrete_model_learner.get_optimistic_model()
    assert safe_precondition is not None
    assert len(safe_precondition.operands) == 0
    assert len(safe_effects) == 0


def test_get_optimistic_model_when_observed_a_single_positive_observation_returns_preconditions_and_effects_based_on_the_observation(
        online_discrete_model_learner: OnlineDiscreteModelLearner, lifted_vocabulary: Set[Predicate]
):
    pre_state_predicates = {
        p for p in lifted_vocabulary if
        p.untyped_representation not in ["(not (available ?x))", "(surface-condition ?x ?oldsurface)", ]
    }
    next_state_predicates = {
        p for p in lifted_vocabulary if
        p.untyped_representation not in ["(available ?x)", "(not (surface-condition ?x ?oldsurface))", ]
    }
    online_discrete_model_learner.add_transition_data(pre_state_predicates, next_state_predicates,
                                                      is_transition_successful=True)
    optimistic_precondition, optimistic_effects = online_discrete_model_learner.get_optimistic_model()
    assert optimistic_precondition is not None
    assert len(optimistic_precondition.operands) == 0
    assert len(optimistic_effects) == len(lifted_vocabulary) - len(next_state_predicates)


def test_get_optimistic_model_when_observed_a_single_negative_observation_returns_precondition_with_or_on_the_must_be_preconditions_cnf_data(
        online_discrete_model_learner: OnlineDiscreteModelLearner, lifted_vocabulary: Set[Predicate]
):
    pre_state_predicates = {
        p for p in lifted_vocabulary if
        p.untyped_representation not in ["(not (available ?x))", "(surface-condition ?x ?oldsurface)", ]
    }
    next_state_predicates = {
        p for p in lifted_vocabulary if
        p.untyped_representation not in ["(available ?x)", "(not (surface-condition ?x ?oldsurface))", ]
    }
    online_discrete_model_learner.add_transition_data(pre_state_predicates, next_state_predicates,
                                                      is_transition_successful=False)
    optimistic_precondition, optimistic_effects = online_discrete_model_learner.get_optimistic_model()
    assert optimistic_precondition is not None
    assert len(optimistic_precondition.operands) == 1
    or_condition = optimistic_precondition.operands.pop()
    assert isinstance(or_condition, Precondition)
    assert or_condition.binary_operator == "or"
    assert len(or_condition.operands) == 2
    assert len(optimistic_effects) > 0
    assert all([eff.is_positive for eff in optimistic_effects])


def test_get_optimistic_model_when_cnfs_are_multiple_unit_clauses_combines_them_in_and_condition(
        online_discrete_model_learner: OnlineDiscreteModelLearner, lifted_vocabulary: Set[Predicate]
):
    for predicate in lifted_vocabulary:
        online_discrete_model_learner.must_be_preconditions.append({predicate})

    optimistic_precondition, _ = online_discrete_model_learner.get_optimistic_model()
    assert optimistic_precondition is not None
    assert len(optimistic_precondition.operands) > 1
    assert all([isinstance(op, Predicate) for op in optimistic_precondition.operands])


def test_is_state_in_safe_model_when_no_observation_was_given_returns_false_since_model_is_conservative(
        online_discrete_model_learner: OnlineDiscreteModelLearner, lifted_vocabulary: Set[Predicate]
):
    state = {p for p in lifted_vocabulary if
             p.untyped_representation not in ["(not (available ?x))", "(surface-condition ?x ?oldsurface)", ]}
    assert not online_discrete_model_learner.is_state_in_safe_model(state)


def test_is_state_in_safe_model_when_predicates_not_are_part_of_what_cannot_be_preconditions_returns_false(
        online_discrete_model_learner: OnlineDiscreteModelLearner, lifted_vocabulary: Set[Predicate]
):
    pre_state_predicates = {
        p for p in lifted_vocabulary if
        p.untyped_representation not in ["(not (available ?x))", "(surface-condition ?x ?oldsurface)", ]
    }
    next_state_predicates = {
        p for p in lifted_vocabulary if
        p.untyped_representation not in ["(available ?x)", "(not (surface-condition ?x ?oldsurface))", ]
    }
    online_discrete_model_learner.add_transition_data(pre_state_predicates, next_state_predicates,
                                                      is_transition_successful=True)
    state = {p for p in lifted_vocabulary if
             p.untyped_representation in ["(not (available ?x))", "(surface-condition ?x ?oldsurface)", ]}
    assert not online_discrete_model_learner.is_state_in_safe_model(state)


def test_is_state_in_safe_model_when_predicates_are_exactly_of_what_cannot_be_preconditions_returns_true(
        online_discrete_model_learner: OnlineDiscreteModelLearner, lifted_vocabulary: Set[Predicate]
):
    pre_state_predicates = {
        p for p in lifted_vocabulary if
        p.untyped_representation not in ["(not (available ?x))", "(surface-condition ?x ?oldsurface)", ]
    }
    next_state_predicates = {
        p for p in lifted_vocabulary if
        p.untyped_representation not in ["(available ?x)", "(not (surface-condition ?x ?oldsurface))", ]
    }
    online_discrete_model_learner.add_transition_data(pre_state_predicates, next_state_predicates,
                                                      is_transition_successful=True)
    state = {p for p in lifted_vocabulary if
             p.untyped_representation not in ["(not (available ?x))", "(surface-condition ?x ?oldsurface)", ]}
    assert online_discrete_model_learner.is_state_in_safe_model(state)


def test_is_state_in_safe_model_when_predicates_are_subset_of_the_inverse_of_what_cannot_be_preconditions_returns_false(
        online_discrete_model_learner: OnlineDiscreteModelLearner, lifted_vocabulary: Set[Predicate]
):
    pre_state_predicates = {
        p for p in lifted_vocabulary if
        p.untyped_representation not in ["(not (available ?x))", "(surface-condition ?x ?oldsurface)", ]
    }
    next_state_predicates = {
        p for p in lifted_vocabulary if
        p.untyped_representation not in ["(available ?x)", "(not (surface-condition ?x ?oldsurface))", ]
    }
    online_discrete_model_learner.add_transition_data(pre_state_predicates, next_state_predicates,
                                                      is_transition_successful=True)
    state = {
        p
        for p in lifted_vocabulary
        if p.untyped_representation
           not in ["(not (available ?x))", "(surface-condition ?x ?oldsurface)", "(available ?x)",
                   "(not (surface-condition ?x ?oldsurface))"]
    }
    assert not online_discrete_model_learner.is_state_in_safe_model(state)


def test_is_state_not_applicable_in_safe_model_when_no_observation_was_given_returns_false(
        online_discrete_model_learner: OnlineDiscreteModelLearner, lifted_vocabulary: Set[Predicate]
):
    state = {p for p in lifted_vocabulary if
             p.untyped_representation not in ["(not (available ?x))", "(surface-condition ?x ?oldsurface)", ]}
    assert not online_discrete_model_learner.is_state_not_applicable_in_safe_discrete_model(state)


def test_is_state_not_applicable_in_safe_model_when_no_nust_be_preconditions_existing_returns_false(
        online_discrete_model_learner: OnlineDiscreteModelLearner, lifted_vocabulary: Set[Predicate]
):
    pre_state_predicates = {
        p for p in lifted_vocabulary if
        p.untyped_representation not in ["(not (available ?x))", "(surface-condition ?x ?oldsurface)", ]
    }
    next_state_predicates = {
        p for p in lifted_vocabulary if
        p.untyped_representation not in ["(available ?x)", "(not (surface-condition ?x ?oldsurface))", ]
    }
    online_discrete_model_learner.add_transition_data(pre_state_predicates, next_state_predicates,
                                                      is_transition_successful=True)
    state = {p for p in lifted_vocabulary if
             p.untyped_representation not in ["(not (available ?x))", "(surface-condition ?x ?oldsurface)", ]}
    assert not online_discrete_model_learner.is_state_not_applicable_in_safe_discrete_model(state)


def test_is_state_not_applicable_in_safe_model_when_state_predicates_are_subset_of_not_preconditions_and_there_are_must_be_preconditions(
        online_discrete_model_learner: OnlineDiscreteModelLearner, lifted_vocabulary: Set[Predicate]
):
    pre_state_predicates = {
        p for p in lifted_vocabulary if
        p.untyped_representation in ["(not (available ?x))", "(surface-condition ?x ?oldsurface)", ]
    }
    next_state_predicates = {
        p for p in lifted_vocabulary if
        p.untyped_representation in ["(available ?x)", "(not (surface-condition ?x ?oldsurface))", ]
    }
    online_discrete_model_learner.add_transition_data(pre_state_predicates, next_state_predicates,
                                                      is_transition_successful=True)
    negative_pre_state_predicates = {
        p for p in lifted_vocabulary if
        p.untyped_representation in ["(available ?x)", "(surface-condition ?x ?oldsurface)", ]
    }
    online_discrete_model_learner._add_negative_pre_state_observation(negative_pre_state_predicates)
    state = {p for p in lifted_vocabulary if
             p.untyped_representation in ["(available ?x)", "(not (surface-condition ?x ?oldsurface))", ]}
    assert online_discrete_model_learner.is_state_not_applicable_in_safe_discrete_model(state)


def test_is_state_not_applicable_in_safe_model_when_state_predicates_do_not_contain_any_unit_clause_of_must_be_preconditions(
        online_discrete_model_learner: OnlineDiscreteModelLearner, lifted_vocabulary: Set[Predicate]
):
    pre_state_predicates = {
        p for p in lifted_vocabulary if
        p.untyped_representation in ["(not (available ?x))", "(surface-condition ?x ?oldsurface)", ]
    }
    next_state_predicates = {
        p for p in lifted_vocabulary if
        p.untyped_representation in ["(available ?x)", "(not (surface-condition ?x ?oldsurface))", ]
    }
    online_discrete_model_learner.add_transition_data(pre_state_predicates, next_state_predicates,
                                                      is_transition_successful=True)
    negative_pre_state_predicates = {
        p for p in lifted_vocabulary if
        p.untyped_representation in ["(available ?x)", "(surface-condition ?x ?oldsurface)"]
    }
    online_discrete_model_learner._add_negative_pre_state_observation(negative_pre_state_predicates)
    state = {p for p in lifted_vocabulary if p.untyped_representation in ["(surface-condition ?x ?oldsurface)", ]}
    assert online_discrete_model_learner.is_state_not_applicable_in_safe_discrete_model(state)
