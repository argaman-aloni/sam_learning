"""Module test for the propositional information gain functionality."""
from typing import Set

import pytest
from pddl_plus_parser.models import Domain, Predicate

from sam_learning.core import OnlineDiscreteModelLearner, VocabularyCreator

TEST_ACTION_NAME = "test_action"


@pytest.fixture
def lifted_vocabulary(woodworking_domain: Domain) -> Set[Predicate]:
    return VocabularyCreator().create_lifted_vocabulary(
        domain=woodworking_domain, possible_parameters=woodworking_domain.actions["do-grind"].signature
    )


@pytest.fixture
def propositional_information_gain_learner(lifted_vocabulary) -> OnlineDiscreteModelLearner:
    return OnlineDiscreteModelLearner(TEST_ACTION_NAME, lifted_vocabulary)


def test_add_positive_sample_when_no_preconditions_exists_adds_it_to_the_superset_creates_a_complementary_set_for_the_not_preconditions_and_returns_none(
    propositional_information_gain_learner: OnlineDiscreteModelLearner, lifted_vocabulary: Set[Predicate]
):
    new_positive_sample = {p for p in lifted_vocabulary if p.untyped_representation in ["(available ?x)", "(treatment ?x ?oldtreatment)"]}
    propositional_information_gain_learner._add_positive_pre_state_observation(new_positive_sample)
    assert len(propositional_information_gain_learner.preconditions_superset) == 2
    assert len(propositional_information_gain_learner.cannot_be_preconditions) == len(lifted_vocabulary) - 2
    assert len(propositional_information_gain_learner.must_be_preconditions) == 0


def test_add_positive_sample_when_sample_already_added_does_not_change_sets(
    propositional_information_gain_learner: OnlineDiscreteModelLearner, lifted_vocabulary: Set[Predicate]
):
    new_positive_sample = {p for p in lifted_vocabulary if p.untyped_representation in ["(available ?x)", "(treatment ?x ?oldtreatment)"]}
    propositional_information_gain_learner._add_positive_pre_state_observation(new_positive_sample)
    assert len(propositional_information_gain_learner.preconditions_superset) == 2
    assert len(propositional_information_gain_learner.cannot_be_preconditions) == len(lifted_vocabulary) - 2
    assert len(propositional_information_gain_learner.must_be_preconditions) == 0
    propositional_information_gain_learner._add_positive_pre_state_observation(new_positive_sample)
    assert len(propositional_information_gain_learner.preconditions_superset) == 2
    assert len(propositional_information_gain_learner.cannot_be_preconditions) == len(lifted_vocabulary) - 2
    assert len(propositional_information_gain_learner.must_be_preconditions) == 0


def test_add_negative_sample_adds_correct_set_of_missing_preconditions_to_the_must_be_preconditions_set(
    propositional_information_gain_learner: OnlineDiscreteModelLearner, lifted_vocabulary: Set[Predicate]
):
    new_negative_sample = {p for p in lifted_vocabulary if p.untyped_representation in ["(not (available ?x))", "(not (treatment ?x ?oldtreatment))"]}
    propositional_information_gain_learner.add_negative_sample(new_negative_sample)
    assert len(propositional_information_gain_learner.must_be_preconditions) == 1
    must_be_pre = propositional_information_gain_learner.must_be_preconditions[0]
    assert len(must_be_pre) == len(lifted_vocabulary) - 2
    assert {p.untyped_representation for p in must_be_pre}.issuperset({"(available ?x)", "(treatment ?x ?oldtreatment)"})


def test_add_negative_and_positive_sample_adds_correct_set_of_missing_preconditions_to_the_must_be_preconditions_set(
    propositional_information_gain_learner: OnlineDiscreteModelLearner, lifted_vocabulary: Set[Predicate]
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
    propositional_information_gain_learner._add_positive_pre_state_observation(new_positive_sample)
    assert len(propositional_information_gain_learner.preconditions_superset) == 6
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
    propositional_information_gain_learner.add_negative_sample(new_negative_sample)
    assert len(propositional_information_gain_learner.must_be_preconditions) == 1
    must_be_pre = propositional_information_gain_learner.must_be_preconditions[0]
    assert len(must_be_pre) == 2
    assert {p.untyped_representation for p in must_be_pre} == {"(available ?x)", "(treatment ?x ?oldtreatment)"}


def test_add_negative_and_positive_sample_removes_redundant_preconditions_from_must_be_preconditions(
    propositional_information_gain_learner: OnlineDiscreteModelLearner, lifted_vocabulary: Set[Predicate]
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
    propositional_information_gain_learner._add_positive_pre_state_observation(new_positive_sample)
    assert len(propositional_information_gain_learner.preconditions_superset) == 7
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
    propositional_information_gain_learner.add_negative_sample(new_negative_sample)
    assert len(propositional_information_gain_learner.must_be_preconditions) == 1
    must_be_pre = propositional_information_gain_learner.must_be_preconditions[0]
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
    propositional_information_gain_learner._add_positive_pre_state_observation(new_positive_sample)
    must_be_pre = propositional_information_gain_learner.must_be_preconditions[0]
    assert len(must_be_pre) == 2


def test_calculate_sample_information_gain_when_not_observed_any_state_yet_returns_value_larger_than_zero(
    propositional_information_gain_learner: OnlineDiscreteModelLearner, lifted_vocabulary: Set[Predicate]
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
    assert propositional_information_gain_learner.calculate_sample_information_gain(new_positive_sample) > 0


def test_calculate_sample_information_gain_when_state_is_superset_of_existing_preconditions_returns_zero(
    propositional_information_gain_learner: OnlineDiscreteModelLearner, lifted_vocabulary: Set[Predicate]
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
    propositional_information_gain_learner._add_positive_pre_state_observation(new_positive_sample)
    tested_sample = {
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
    assert propositional_information_gain_learner.calculate_sample_information_gain(tested_sample) == 0


def test_calculate_sample_information_gain_when_state_is_subset_of_negative_samples_returns_zero(
    propositional_information_gain_learner: OnlineDiscreteModelLearner, lifted_vocabulary: Set[Predicate]
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
    propositional_information_gain_learner._add_positive_pre_state_observation(new_positive_sample)
    tested_sample = {p for p in lifted_vocabulary if p.untyped_representation in ["(not (available ?x))", "(not (surface-condition ?x ?oldsurface))"]}
    assert propositional_information_gain_learner.calculate_sample_information_gain(tested_sample) == 0
