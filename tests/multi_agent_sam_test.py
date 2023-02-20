"""Module test for the multi agent action model learning."""
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser
from pddl_plus_parser.models import Domain, Problem, MultiAgentObservation, ActionCall, MultiAgentComponent, \
    JointActionCall, NOP_ACTION, GroundedPredicate, Predicate
from pytest import fixture

from sam_learning.core import LiteralCNF
from sam_learning.learners import MultiAgentSAM
from tests.consts import WOODWORKING_COMBINED_DOMAIN_PATH, WOODWORKING_COMBINED_PROBLEM_PATH, \
    WOODWORKING_COMBINED_TRAJECTORY_PATH, ROVERS_COMBINED_DOMAIN_PATH, ROVERS_COMBINED_PROBLEM_PATH, \
    ROVERS_COMBINED_TRAJECTORY_PATH

WOODWORKING_AGENT_NAMES = ["glazer0", "grinder0", "highspeed-saw0", "immersion-varnisher0", "planer0", "saw0",
                           "spray-varnisher0"]
ROVERS_AGENT_NAMES = [f"rovers{i}" for i in range(10)]


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


@fixture()
def rovers_domain() -> Domain:
    return DomainParser(ROVERS_COMBINED_DOMAIN_PATH, partial_parsing=True).parse_domain()


@fixture()
def rovers_problem(rovers_domain: Domain) -> Problem:
    return ProblemParser(problem_path=ROVERS_COMBINED_PROBLEM_PATH, domain=rovers_domain).parse_problem()


@fixture()
def rovers_ma_observation(rovers_domain: Domain, rovers_problem: Problem) -> MultiAgentObservation:
    return TrajectoryParser(rovers_domain, rovers_problem).parse_trajectory(
        ROVERS_COMBINED_TRAJECTORY_PATH, executing_agents=ROVERS_AGENT_NAMES)


@fixture()
def rovers_ma_sam(rovers_domain: Domain) -> MultiAgentSAM:
    return MultiAgentSAM(rovers_domain)


@fixture()
def ma_sam(combined_domain: Domain) -> MultiAgentSAM:
    return MultiAgentSAM(combined_domain)


def test_initialize_cnfs_sets_correct_predicates_in_the_cnf_dictionary(ma_sam: MultiAgentSAM, combined_domain: Domain):
    ma_sam._initialize_cnfs()
    assert len(ma_sam.literals_cnf) == 2 * len(combined_domain.predicates)


def test_initialize_cnfs_sets_negative_predicates_correctly_in_the_negative_cnf(ma_sam: MultiAgentSAM,
                                                                                combined_domain: Domain):
    ma_sam._initialize_cnfs()
    positive_literals = [literal for literal in ma_sam.literals_cnf if not literal.startswith("(not ")]
    negative_literals = [literal for literal in ma_sam.literals_cnf if literal.startswith("(not ")]
    assert len(positive_literals) == len(negative_literals)
    assert len(positive_literals) == len(combined_domain.predicates)


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
    ma_sam.literals_cnf["(surface-condition ?obj ?surface)"] = ma_literals_cnf
    possible_effects = [("do-immersion-varnish", "(surface-condition ?agent ?newcolour)"),
                        ("do-grind", "(surface-condition ?agent ?oldcolour)"),
                        ("do-plane", "(surface-condition ?agent ?colour)")]
    ma_literals_cnf.add_possible_effect(possible_effects)
    predicate_params = ["?agent", "?colour"]
    predicate_types = combined_domain.predicates["surface-condition"].signature.values()
    ma_sam.observed_actions.append("do-plane")
    ma_sam.construct_safe_actions()

    assert "do-plane" not in ma_sam.safe_actions


def test_update_single_agent_executed_action_updates_action_count(
        ma_sam: MultiAgentSAM, multi_agent_observation: MultiAgentObservation):
    first_trajectory_component = multi_agent_observation.components[0]
    test_action = ActionCall("do-grind", ["grinder0", "p0", "smooth", "red", "varnished", "colourfragments"])
    ma_sam._initialize_cnfs()
    ma_sam._create_fully_observable_triplet_predicates(
        test_action,
        first_trajectory_component.previous_state,
        first_trajectory_component.next_state)

    ma_sam.update_single_agent_executed_action(
        executed_action=test_action,
        previous_state=first_trajectory_component.previous_state,
        next_state=first_trajectory_component.next_state)
    assert "do-grind" in ma_sam.observed_actions


def test_construct_safe_actions_returns_safe_action_when_it_has_only_one_effect_with_no_ambiguities(
        ma_sam: MultiAgentSAM, do_plane_first_action_call: ActionCall, combined_domain: Domain,
        ma_literals_cnf: LiteralCNF):
    ma_sam.literals_cnf["(surface-condition ?obj ?surface)"] = ma_literals_cnf
    possible_effects = [("do-grind", "(surface-condition ?m ?oldcolour)"),
                        ("do-plane", "(surface-condition ?m ?colour)")]
    ma_literals_cnf.add_possible_effect(possible_effects)
    ma_literals_cnf.add_possible_effect([("do-immersion-varnish", "(surface-condition ?m ?newcolour)")])
    ma_sam.observed_actions.append("do-plane")
    ma_sam.observed_actions.append("do-immersion-varnish")
    ma_sam.observed_actions.append("do-grind")

    ma_sam.construct_safe_actions()
    assert "do-plane" not in ma_sam.safe_actions
    assert "do-grind" not in ma_sam.safe_actions
    assert "do-immersion-varnish" in ma_sam.safe_actions


def test_learn_action_model_with_colliding_actions_returns_that_actions_are_unsafe(
        rovers_ma_sam: MultiAgentSAM, rovers_ma_observation: MultiAgentObservation):
    _, learning_report = rovers_ma_sam.learn_combined_action_model([rovers_ma_observation])
    assert learning_report["navigate"] == "NOT SAFE"
    assert learning_report["communicate_rock_data"] == "NOT SAFE"


def test_learn_action_model_returns_learned_model(
        ma_sam: MultiAgentSAM, multi_agent_observation: MultiAgentObservation):
    learned_model, learning_report = ma_sam.learn_combined_action_model([multi_agent_observation])
    print(learning_report)
    print(learned_model.to_pddl())
