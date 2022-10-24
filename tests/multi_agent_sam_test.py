"""Module test for the multi agent action model learning."""
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser
from pddl_plus_parser.models import Domain, Problem, MultiAgentObservation, ActionCall, MultiAgentComponent, \
    JointActionCall, NOP_ACTION, GroundedPredicate, Predicate
from pytest import fixture

from sam_learning.core import LiteralCNF
from sam_learning.learners import MultiAgentSAM
from tests.consts import WOODWORKING_COMBINED_DOMAIN_PATH, WOODWORKING_COMBINED_PROBLEM_PATH, \
    WOODWORKING_COMBINED_TRAJECTORY_PATH

WOODWORKING_AGENT_NAMES = ["glazer0", "grinder0", "highspeed-saw0", "immersion-varnisher0", "planer0", "saw0",
                           "spray-varnisher0"]


@fixture()
def combined_domain() -> Domain:
    return DomainParser(WOODWORKING_COMBINED_DOMAIN_PATH, partial_parsing=True).parse_domain()


@fixture()
def combined_problem(combined_domain: Domain) -> Problem:
    return ProblemParser(problem_path=WOODWORKING_COMBINED_PROBLEM_PATH, domain=combined_domain).parse_problem()


@fixture()
def multi_agent_observation(combined_domain: Domain, combined_problem: Problem) -> MultiAgentObservation:
    return TrajectoryParser(combined_domain, combined_problem).parse_trajectory(
        WOODWORKING_COMBINED_TRAJECTORY_PATH, executing_agents=WOODWORKING_AGENT_NAMES)


@fixture()
def ma_sam(combined_domain: Domain) -> MultiAgentSAM:
    return MultiAgentSAM(combined_domain)


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
def ma_literals_cnf(combined_domain: Domain) -> LiteralCNF:
    action_names = [action for action in combined_domain.actions.keys()]
    return LiteralCNF(action_names)


def test_initialize_cnfs_sets_correct_predicates_in_the_cnf_dictionary(ma_sam: MultiAgentSAM, combined_domain: Domain):
    ma_sam._initialize_cnfs()
    assert len(ma_sam.positive_literals_cnf) == len(combined_domain.predicates)
    assert len(ma_sam.negative_literals_cnf) == len(combined_domain.predicates)


def test_locate_executing_action_when_only_one_action_is_in_the_joint_action_returns_single_action(
        ma_sam: MultiAgentSAM, do_plane_first_action_call: ActionCall):
    assert ma_sam._locate_executing_action(JointActionCall(
        [do_plane_first_action_call, ActionCall(NOP_ACTION, [])])) == [do_plane_first_action_call]


def test_locate_executing_action_when_two_actions_are_in_the_joint_action_returns_two_actions(
        ma_sam: MultiAgentSAM, do_plane_first_action_call: ActionCall, do_plane_second_action_call: ActionCall):
    assert ma_sam._locate_executing_action(JointActionCall(
        [do_plane_first_action_call, do_plane_second_action_call, ActionCall(NOP_ACTION, [])])) == \
           [do_plane_first_action_call, do_plane_second_action_call]


def test_create_fully_observable_predicates_adds_all_missing_state_predicates_correctly(
        ma_sam: MultiAgentSAM, multi_agent_observation: MultiAgentObservation, do_plane_first_action_call: ActionCall):
    component = multi_agent_observation.components[1]
    next_state = component.next_state
    observed_objects = multi_agent_observation.grounded_objects
    positive_predicates, negative_predicates = ma_sam.create_fully_observable_predicates(next_state, observed_objects)
    state_predicates = []
    for predicates in next_state.state_predicates.values():
        state_predicates.extend(predicates)

    assert len(positive_predicates) == len(state_predicates)
    assert len(negative_predicates) > 0
    negative_predicates_str = [p.untyped_representation for p in negative_predicates]
    positive_predicates_str = [p.untyped_representation for p in positive_predicates]
    assert all([p not in positive_predicates_str for p in negative_predicates_str])


def test_compute_interacting_actions_returns_empty_list_if_no_action_interacts_with_the_predicate(
        ma_sam: MultiAgentSAM, do_plane_first_action_call: ActionCall, combined_domain: Domain):
    lifted_predicate = combined_domain.predicates["boardsize-successor"]
    grounded_predicate = GroundedPredicate(name="boardsize-successor", signature=lifted_predicate.signature,
                                           object_mapping={"?size1": "s0", "?size2": "s1"})
    assert ma_sam.compute_interacting_actions(grounded_predicate, executing_actions=[do_plane_first_action_call]) == []


def test_compute_interacting_actions_returns_one_action_call_if_only_one_interacts_with_the_predicate(
        ma_sam: MultiAgentSAM, do_plane_first_action_call: ActionCall, combined_domain: Domain):
    # (do-plane planer0 p2 verysmooth natural varnished)
    lifted_predicate = combined_domain.predicates["surface-condition"]
    grounded_predicate = GroundedPredicate(name="surface-condition", signature=lifted_predicate.signature,
                                           object_mapping={"?obj": "p2", "?surface": "verysmooth"})
    assert ma_sam.compute_interacting_actions(grounded_predicate, executing_actions=[do_plane_first_action_call]) == \
           [do_plane_first_action_call]


def test_compute_interacting_actions_returns_two_actions_when_two_actions_interact_with_the_same_predicate(
        ma_sam: MultiAgentSAM, do_plane_first_action_call: ActionCall, do_plane_second_action_call: ActionCall,
        combined_domain: Domain):
    lifted_predicate = combined_domain.predicates["surface-condition"]
    grounded_predicate = GroundedPredicate(name="surface-condition", signature=lifted_predicate.signature,
                                           object_mapping={"?obj": "p2", "?surface": "verysmooth"})

    assert ma_sam.compute_interacting_actions(
        grounded_predicate, executing_actions=[do_plane_first_action_call, do_plane_second_action_call]) == \
           [do_plane_first_action_call, do_plane_second_action_call]


def test_compute_interacting_actions_returns_two_actions_when_two_when_third_action_does_not_interact_with_the_predicate(
        ma_sam: MultiAgentSAM, do_plane_first_action_call: ActionCall, do_plane_second_action_call: ActionCall,
        combined_domain: Domain):
    lifted_predicate = combined_domain.predicates["surface-condition"]
    grounded_predicate = GroundedPredicate(name="surface-condition", signature=lifted_predicate.signature,
                                           object_mapping={"?obj": "p2", "?surface": "verysmooth"})

    non_interacting_action = ActionCall("do-plane", ["planer1", "p1", "verysmooth", "natural", "untreated"])
    assert ma_sam.compute_interacting_actions(
        grounded_predicate,
        executing_actions=[do_plane_first_action_call, do_plane_second_action_call, non_interacting_action]) == \
           [do_plane_first_action_call, do_plane_second_action_call]


def test_construct_safe_actions_returns_empty_list_if_no_action_is_safe(
        ma_sam: MultiAgentSAM, do_plane_first_action_call: ActionCall, combined_domain: Domain,
        ma_literals_cnf: LiteralCNF):
    ma_sam.positive_literals_cnf["(surface-condition ?obj ?surface)"] = ma_literals_cnf
    possible_effects = [("do-immersion-varnish", "(surface-condition ?agent ?newcolour)"),
                        ("do-grind", "(surface-condition ?agent ?oldcolour)"),
                        ("do-plane", "(surface-condition ?agent ?colour)")]
    ma_literals_cnf.add_possible_effect(possible_effects)
    predicate_params = ["?agent", "?colour"]
    predicate_types = combined_domain.predicates["surface-condition"].signature.values()
    ma_sam.lifted_bounded_predicates["do-plane"] = {
        combined_domain.predicates["surface-condition"].untyped_representation:
            {("(surface-condition ?agent ?colour)",
              Predicate("surface-condition", signature={
                  bounded_param: obj_type for bounded_param, obj_type in zip(predicate_params, predicate_types)}))}}
    ma_sam.observed_actions.append("do-plane")
    ma_sam.construct_safe_actions()
    assert "do-plane" not in ma_sam.safe_actions


def test_construct_safe_actions_returns_safe_action_when_it_has_only_one_effect_with_no_ambiguities(
        ma_sam: MultiAgentSAM, do_plane_first_action_call: ActionCall, combined_domain: Domain,
        ma_literals_cnf: LiteralCNF):
    ma_sam.positive_literals_cnf["(surface-condition ?obj ?surface)"] = ma_literals_cnf
    possible_effects = [("do-grind", "(surface-condition ?agent ?oldcolour)"),
                        ("do-plane", "(surface-condition ?agent ?colour)")]
    ma_literals_cnf.add_possible_effect(possible_effects)
    ma_literals_cnf.add_possible_effect([("do-immersion-varnish", "(surface-condition ?agent ?newcolour)")])
    predicate_types = combined_domain.predicates["surface-condition"].signature.values()

    ma_sam.lifted_bounded_predicates["do-plane"] = {
        combined_domain.predicates["surface-condition"].untyped_representation:
            {("(surface-condition ?agent ?colour)",
              Predicate("surface-condition", signature={
                  bounded_param: obj_type for bounded_param, obj_type in
                  zip(["?agent", "?colour"], predicate_types)}))}}

    ma_sam.lifted_bounded_predicates["do-immersion-varnish"] = {
        combined_domain.predicates["surface-condition"].untyped_representation:
            {("(surface-condition ?agent ?newcolour)",
              Predicate("surface-condition", signature={
                  bounded_param: obj_type for bounded_param, obj_type in
                  zip(["?agent", "?newcolour"], predicate_types)}))}}

    ma_sam.lifted_bounded_predicates["do-grind"] = {
        combined_domain.predicates["surface-condition"].untyped_representation:
            {("(surface-condition ?agent ?oldcolour)",
              Predicate("surface-condition", signature={
                  bounded_param: obj_type for bounded_param, obj_type in
                  zip(["?agent", "?oldcolour"], predicate_types)}))}}

    ma_sam.observed_actions.append("do-plane")
    ma_sam.observed_actions.append("do-immersion-varnish")
    ma_sam.observed_actions.append("do-grind")

    ma_sam.construct_safe_actions()
    assert "do-plane" not in ma_sam.safe_actions
    assert "do-grind" not in ma_sam.safe_actions
    assert "do-immersion-varnish" in ma_sam.safe_actions


def test_learn_action_model_returns_learned_model(
        ma_sam: MultiAgentSAM, multi_agent_observation: MultiAgentObservation):
    learned_model, learning_report = ma_sam.learn_combined_action_model([multi_agent_observation])
    print(learning_report)
    print(learned_model.to_pddl())
