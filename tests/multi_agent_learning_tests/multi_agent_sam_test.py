"""Module test for the multi-agent action model learning."""
from pddl_plus_parser.models import Domain, MultiAgentObservation, ActionCall, MultiAgentComponent, GroundedPredicate, Predicate
from pytest import fixture

from sam_learning.core import LiteralCNF
from sam_learning.learners import MultiAgentSAM
from tests.consts import sync_ma_snapshot
from utilities import NegativePreconditionPolicy

WOODWORKING_AGENT_NAMES = ["glazer0", "grinder0", "highspeed-saw0", "immersion-varnisher0", "planer0", "saw0", "spray-varnisher0"]
ROVERS_AGENT_NAMES = [f"rovers{i}" for i in range(10)]


@fixture()
def woodworking_ma_sam(woodworking_ma_combined_domain: Domain) -> MultiAgentSAM:
    return MultiAgentSAM(woodworking_ma_combined_domain)


@fixture()
def woodworking_ma_sam_hard_policy(woodworking_ma_combined_domain: Domain) -> MultiAgentSAM:
    return MultiAgentSAM(woodworking_ma_combined_domain, negative_precondition_policy=NegativePreconditionPolicy.hard)


@fixture()
def woodworking_ma_sam_soft_policy(woodworking_ma_combined_domain: Domain) -> MultiAgentSAM:
    return MultiAgentSAM(woodworking_ma_combined_domain, negative_precondition_policy=NegativePreconditionPolicy.soft)


@fixture()
def rovers_ma_sam(ma_rovers_domain) -> MultiAgentSAM:
    return MultiAgentSAM(ma_rovers_domain)


@fixture()
def do_plane_observation_component(multi_agent_observation: MultiAgentObservation) -> MultiAgentComponent:
    return multi_agent_observation.components[1]


@fixture()
def do_plane_first_action_call() -> ActionCall:
    return ActionCall("do-plane", ["planer0", "p2", "verysmooth", "natural", "varnished"])


@fixture()
def do_plane_second_action_call() -> ActionCall:
    return ActionCall("do-plane", ["planer1", "p2", "verysmooth", "natural", "untreated"])


@fixture()
def communicate_image_data_action_call() -> ActionCall:
    return ActionCall("communicate_image_data", ["rover0", "lander0", "objective4", "colour", "waypoint0", "waypoint1"])


@fixture()
def woodworking_literals_cnf(woodworking_ma_combined_domain: Domain) -> LiteralCNF:
    action_names = [action for action in woodworking_ma_combined_domain.actions.keys()]
    return LiteralCNF(action_names)


@fixture()
def rovers_literals_cnf(ma_rovers_domain: Domain) -> LiteralCNF:
    action_names = [action for action in ma_rovers_domain.actions.keys()]
    return LiteralCNF(action_names)


def test_initialize_cnfs_sets_correct_predicates_in_the_cnf_dictionary(woodworking_ma_sam: MultiAgentSAM, woodworking_ma_combined_domain: Domain):
    woodworking_ma_sam._initialize_cnfs()
    assert len(woodworking_ma_sam.literals_cnf) == 2 * len(woodworking_ma_combined_domain.predicates)


def test_initialize_cnfs_sets_negative_predicates_correctly_in_the_negative_cnf(
    woodworking_ma_sam: MultiAgentSAM, woodworking_ma_combined_domain: Domain
):
    woodworking_ma_sam._initialize_cnfs()
    positive_literals = [literal for literal in woodworking_ma_sam.literals_cnf if not literal.startswith("(not ")]
    negative_literals = [literal for literal in woodworking_ma_sam.literals_cnf if literal.startswith("(not ")]
    assert len(positive_literals) == len(negative_literals)
    assert len(negative_literals) == len(woodworking_ma_combined_domain.predicates)


def test_extract_relevant_not_effects_returns_empty_list_if_none_of_the_predicates_is_relevant_to_the_action(
    woodworking_ma_sam: MultiAgentSAM, woodworking_ma_combined_domain: Domain, do_plane_first_action_call: ActionCall
):
    lifted_predicate = woodworking_ma_combined_domain.predicates["boardsize-successor"]
    in_state_test_predicates = {
        GroundedPredicate(name="boardsize-successor", signature=lifted_predicate.signature, object_mapping={"?size1": "s8", "?size2": "s10"})
    }
    assert (
        woodworking_ma_sam._extract_relevant_not_effects(
            in_state_predicates=in_state_test_predicates,
            removed_state_predicates=set(),
            executing_actions=[do_plane_first_action_call],
            relevant_action=do_plane_first_action_call,
        )
        == []
    )


def test_extract_relevant_not_effects_returns_one_negative_literal_when_only_giving_literals_in_state(
    woodworking_ma_sam: MultiAgentSAM, woodworking_ma_combined_domain: Domain, do_plane_first_action_call: ActionCall
):
    lifted_predicate = woodworking_ma_combined_domain.predicates["surface-condition"]
    in_state_test_predicates = {
        GroundedPredicate(name="surface-condition", signature=lifted_predicate.signature, object_mapping={"?obj": "p2", "?surface": "verysmooth"})
    }
    not_effects = woodworking_ma_sam._extract_relevant_not_effects(
        in_state_predicates=in_state_test_predicates,
        removed_state_predicates=set(),
        executing_actions=[do_plane_first_action_call],
        relevant_action=do_plane_first_action_call,
    )
    assert len(not_effects) == 1
    assert not_effects[0].untyped_representation == "(not (surface-condition p2 verysmooth))"


def test_extract_relevant_not_effects_returns_one_positive_literal_when_only_giving_literals_in_state(
    woodworking_ma_sam: MultiAgentSAM, woodworking_ma_combined_domain: Domain, do_plane_first_action_call: ActionCall
):
    lifted_predicate = woodworking_ma_combined_domain.predicates["surface-condition"]
    removed_state_test_predicates = {
        GroundedPredicate(
            name="surface-condition", signature=lifted_predicate.signature, object_mapping={"?obj": "p2", "?surface": "verysmooth"}, is_positive=False
        )
    }
    not_effects = woodworking_ma_sam._extract_relevant_not_effects(
        in_state_predicates=set(),
        removed_state_predicates=removed_state_test_predicates,
        executing_actions=[do_plane_first_action_call],
        relevant_action=do_plane_first_action_call,
    )

    assert len(not_effects) == 1
    assert not_effects[0].untyped_representation == "(surface-condition p2 verysmooth)"


def test_compute_interacting_actions_returns_empty_list_if_no_action_interacts_with_the_predicate(
    woodworking_ma_sam: MultiAgentSAM, do_plane_first_action_call: ActionCall, woodworking_ma_combined_domain: Domain
):
    lifted_predicate = woodworking_ma_combined_domain.predicates["boardsize-successor"]
    grounded_predicate = GroundedPredicate(
        name="boardsize-successor", signature=lifted_predicate.signature, object_mapping={"?size1": "s0", "?size2": "s1"}
    )
    assert woodworking_ma_sam.compute_interacting_actions(grounded_predicate, executing_actions=[do_plane_first_action_call]) == []


def test_compute_interacting_actions_returns_one_action_call_if_only_one_interacts_with_the_predicate(
    woodworking_ma_sam: MultiAgentSAM, do_plane_first_action_call: ActionCall, woodworking_ma_combined_domain: Domain
):
    # (do-plane planer0 p2 verysmooth natural varnished)
    lifted_predicate = woodworking_ma_combined_domain.predicates["surface-condition"]
    grounded_predicate = GroundedPredicate(
        name="surface-condition", signature=lifted_predicate.signature, object_mapping={"?obj": "p2", "?surface": "verysmooth"}
    )
    assert woodworking_ma_sam.compute_interacting_actions(grounded_predicate, executing_actions=[do_plane_first_action_call]) == [
        do_plane_first_action_call
    ]


def test_compute_interacting_actions_returns_two_actions_when_two_actions_interact_with_the_same_predicate(
    woodworking_ma_sam: MultiAgentSAM,
    do_plane_first_action_call: ActionCall,
    do_plane_second_action_call: ActionCall,
    woodworking_ma_combined_domain: Domain,
):
    lifted_predicate = woodworking_ma_combined_domain.predicates["surface-condition"]
    grounded_predicate = GroundedPredicate(
        name="surface-condition", signature=lifted_predicate.signature, object_mapping={"?obj": "p2", "?surface": "verysmooth"}
    )

    assert woodworking_ma_sam.compute_interacting_actions(
        grounded_predicate, executing_actions=[do_plane_first_action_call, do_plane_second_action_call]
    ) == [do_plane_first_action_call, do_plane_second_action_call]


def test_compute_interacting_actions_returns_two_actions_when_two_when_third_action_does_not_interact_with_the_predicate(
    woodworking_ma_sam: MultiAgentSAM,
    do_plane_first_action_call: ActionCall,
    do_plane_second_action_call: ActionCall,
    woodworking_ma_combined_domain: Domain,
):
    lifted_predicate = woodworking_ma_combined_domain.predicates["surface-condition"]
    grounded_predicate = GroundedPredicate(
        name="surface-condition", signature=lifted_predicate.signature, object_mapping={"?obj": "p2", "?surface": "verysmooth"}
    )

    non_interacting_action = ActionCall("do-plane", ["planer1", "p1", "verysmooth", "natural", "untreated"])
    assert woodworking_ma_sam.compute_interacting_actions(
        grounded_predicate, executing_actions=[do_plane_first_action_call, do_plane_second_action_call, non_interacting_action]
    ) == [do_plane_first_action_call, do_plane_second_action_call]


def test_add_not_effect_to_cnf_adds_not_effect_to_cnf_if_not_effect_is_not_in_cnf(
    rovers_ma_sam: MultiAgentSAM, communicate_image_data_action_call: ActionCall, ma_rovers_domain: Domain, rovers_literals_cnf: LiteralCNF
):
    lifted_predicate = ma_rovers_domain.predicates["communicated_image_data"]
    rovers_ma_sam.literals_cnf["(communicated_image_data ?o ?m)"] = rovers_literals_cnf
    not_effects = [
        GroundedPredicate(name="communicated_image_data", signature=lifted_predicate.signature, object_mapping={"?o": "objective4", "?m": "colour"})
    ]

    rovers_ma_sam.add_not_effect_to_cnf(executed_action=communicate_image_data_action_call, not_effects=not_effects)
    assert rovers_ma_sam.literals_cnf["(communicated_image_data ?o ?m)"].not_effects[communicate_image_data_action_call.name] == {
        "(communicated_image_data ?o ?m)"
    }


def test_add_must_be_effect_to_cnf_adds_the_literal_to_possible_effects_when_there_is_an_injective_match(
    rovers_ma_sam: MultiAgentSAM, communicate_image_data_action_call: ActionCall, ma_rovers_domain: Domain, rovers_literals_cnf: LiteralCNF
):
    lifted_predicate = ma_rovers_domain.predicates["communicated_image_data"]
    rovers_ma_sam.literals_cnf["(communicated_image_data ?o ?m)"] = rovers_literals_cnf
    not_effects = {
        GroundedPredicate(name="communicated_image_data", signature=lifted_predicate.signature, object_mapping={"?o": "objective4", "?m": "colour"})
    }

    rovers_ma_sam.add_must_be_effect_to_cnf(executed_action=communicate_image_data_action_call, grounded_effects=not_effects)
    assert rovers_ma_sam.literals_cnf["(communicated_image_data ?o ?m)"].possible_lifted_effects == [
        [(communicate_image_data_action_call.name, "(communicated_image_data ?o ?m)")]
    ]


def test_construct_safe_actions_returns_empty_list_if_no_action_is_safe(
    woodworking_ma_sam: MultiAgentSAM, do_plane_first_action_call: ActionCall, woodworking_ma_combined_domain: Domain, woodworking_literals_cnf
):
    woodworking_ma_sam.literals_cnf["(surface-condition ?obj ?surface)"] = woodworking_literals_cnf
    possible_effects = [
        ("do-immersion-varnish", "(surface-condition ?agent ?newcolour)"),
        ("do-grind", "(surface-condition ?agent ?oldcolour)"),
        ("do-plane", "(surface-condition ?agent ?colour)"),
    ]
    woodworking_literals_cnf.add_possible_effect(possible_effects)
    woodworking_ma_sam.observed_actions.append("do-plane")
    woodworking_ma_sam.construct_safe_actions()
    assert "do-plane" not in woodworking_ma_sam.safe_actions


def test_update_single_agent_executed_action_updates_action_count(woodworking_ma_sam: MultiAgentSAM, multi_agent_observation: MultiAgentObservation):
    first_trajectory_component = multi_agent_observation.components[0]
    test_action = ActionCall("do-grind", ["grinder0", "p0", "smooth", "red", "varnished", "colourfragments"])
    woodworking_ma_sam._initialize_cnfs()
    sync_ma_snapshot(
        ma_sam=woodworking_ma_sam,
        component=first_trajectory_component,
        action_call=test_action,
        trajectory_objects=multi_agent_observation.grounded_objects,
    )
    woodworking_ma_sam.update_single_agent_executed_action(
        executed_action=test_action, previous_state=first_trajectory_component.previous_state, next_state=first_trajectory_component.next_state
    )
    assert "do-grind" in woodworking_ma_sam.observed_actions


def test_construct_safe_actions_returns_safe_action_when_it_has_only_one_effect_with_no_ambiguities(
    woodworking_ma_sam: MultiAgentSAM, do_plane_first_action_call: ActionCall, woodworking_ma_combined_domain: Domain, woodworking_literals_cnf
):
    woodworking_ma_sam.literals_cnf["(surface-condition ?obj ?surface)"] = woodworking_literals_cnf
    possible_effects = [("do-grind", "(surface-condition ?m ?oldcolour)"), ("do-plane", "(surface-condition ?m ?colour)")]
    woodworking_literals_cnf.add_possible_effect(possible_effects)
    woodworking_literals_cnf.add_possible_effect([("do-immersion-varnish", "(surface-condition ?m ?newcolour)")])
    woodworking_ma_sam.observed_actions.append("do-plane")
    woodworking_ma_sam.observed_actions.append("do-immersion-varnish")
    woodworking_ma_sam.observed_actions.append("do-grind")

    woodworking_ma_sam.construct_safe_actions()
    assert "do-plane" not in woodworking_ma_sam.safe_actions
    assert "do-grind" not in woodworking_ma_sam.safe_actions
    assert "do-immersion-varnish" in woodworking_ma_sam.safe_actions


def test_learn_action_model_with_colliding_actions_not_learn_actions_that_are_considered_as_unsafe(
    rovers_ma_sam: MultiAgentSAM, ma_rovers_observation
):
    _, learning_report = rovers_ma_sam.learn_combined_action_model([ma_rovers_observation])
    assert "navigate" not in learning_report
    assert "communicate_rock_data" not in learning_report


def test_learn_action_model_returns_learned_model(woodworking_ma_sam: MultiAgentSAM, multi_agent_observation: MultiAgentObservation):
    learned_model, learning_report = woodworking_ma_sam.learn_combined_action_model([multi_agent_observation])
    print(learning_report)
    print(learned_model.to_pddl())


def test_learn_ma_action_model_with_hard_policy_deletes_negative_preconditions(
    woodworking_ma_sam_hard_policy: MultiAgentSAM, multi_agent_observation: MultiAgentObservation
):
    learned_model_ignore, _ = woodworking_ma_sam_hard_policy.learn_combined_action_model([multi_agent_observation])

    for action in learned_model_ignore.actions.values():
        for pre in action.preconditions.root.operands:
            if isinstance(pre, Predicate):
                assert pre.is_positive


def test_learn_action_model_with_hard_negative_precondition_policy_keep_positive_preconditions(
    woodworking_ma_sam: MultiAgentSAM, multi_agent_observation: MultiAgentObservation, woodworking_ma_sam_hard_policy: MultiAgentSAM
):
    learned_model, _ = woodworking_ma_sam.learn_combined_action_model([multi_agent_observation])
    learned_model_ignore, _ = woodworking_ma_sam_hard_policy.learn_combined_action_model([multi_agent_observation])

    for action, action_ignored in zip(learned_model.actions.values(), learned_model_ignore.actions.values()):
        preconds = {prec for prec in action.preconditions.root.operands if isinstance(prec, Predicate)}
        preconds_ignore = {prec for prec in action_ignored.preconditions.root.operands if isinstance(prec, Predicate)}

        difference_ignore_from_classic = preconds_ignore.difference(preconds)
        difference_classic_from_ignore = preconds.difference(preconds_ignore)

        assert len(difference_ignore_from_classic) == 0

        for pre in difference_classic_from_ignore:
            if isinstance(pre, Predicate):
                assert not pre.is_positive


def test_learn_action_model_with_hard_policy_delete_effect_has_positive_precondition(
    woodworking_ma_sam_hard_policy: MultiAgentSAM, multi_agent_observation: MultiAgentObservation
):
    learned_model, _ = woodworking_ma_sam_hard_policy.learn_combined_action_model([multi_agent_observation])

    for action in learned_model.actions.values():
        predicates = [pre.untyped_representation for pre in action.preconditions.root.operands if pre.is_positive and isinstance(pre, Predicate)]

        del_effects = [eff for eff in action.discrete_effects if not eff.is_positive and isinstance(eff, Predicate)]

        for del_eff in del_effects:
            del_eff.is_positive = not del_eff.is_positive

        flipped_del_effects = {del_eff.untyped_representation for del_eff in del_effects}

        assert len(del_effects) == len(flipped_del_effects.intersection(predicates))


def test_learn_action_model_with_soft_policy_delete_preconditions_has_add_effect(
    woodworking_ma_sam: MultiAgentSAM, woodworking_ma_sam_soft_policy: MultiAgentSAM, multi_agent_observation: MultiAgentObservation
):
    learned_model, _ = woodworking_ma_sam.learn_combined_action_model([multi_agent_observation])
    learned_model_soft, _ = woodworking_ma_sam_soft_policy.learn_combined_action_model([multi_agent_observation])

    for action, action_ignored in zip(learned_model.actions.values(), learned_model_soft.actions.values()):
        preconds = {prec for prec in action.preconditions.root.operands if isinstance(prec, Predicate)}
        preconds_ignore = {prec for prec in action_ignored.preconditions.root.operands if isinstance(prec, Predicate)}

        difference_ignore_from_classic = preconds_ignore.difference(preconds)
        difference_classic_from_ignore = preconds.difference(preconds_ignore)
        add_effects = [effect.untyped_representation for effect in action.discrete_effects if effect.is_positive]

        assert len(difference_ignore_from_classic) == 0

        for pre in difference_classic_from_ignore:
            if isinstance(pre, Predicate):
                pre_positive_copy = pre.copy(is_negated=True)
                assert (not pre.is_positive) and (pre_positive_copy.untyped_representation in add_effects)


def test_learn_action_model_hard_policy_returns_learned_model(
    woodworking_ma_sam_hard_policy: MultiAgentSAM, multi_agent_observation: MultiAgentObservation
):
    learned_model, learning_report = woodworking_ma_sam_hard_policy.learn_combined_action_model([multi_agent_observation])
    print(learning_report)
    print(learned_model.to_pddl())


def test_learn_action_model_soft_policy_returns_learned_model(
    woodworking_ma_sam_soft_policy: MultiAgentSAM, multi_agent_observation: MultiAgentObservation
):
    learned_model, learning_report = woodworking_ma_sam_soft_policy.learn_combined_action_model([multi_agent_observation])
    print(learning_report)
    print(learned_model.to_pddl())
