"""Module test for the multi-agent action model learning."""
import json
from typing import List, Dict

from pddl_plus_parser.lisp_parsers import DomainParser, TrajectoryParser
from pddl_plus_parser.models import Domain, MultiAgentObservation
from pytest import fixture

from sam_learning.learners.multi_agent_numeric_sam import NumericMultiAgentSAM
from tests.consts import STAR_CRAFT_DOMAIN_PATH, STAR_CRAFT_TRAJECTORY_PATH, \
    STAR_CRAFT_FLUENTS_MAP_PATH

STARCRAFT_AGENT_NAMES = [f"agent{i}" for i in range(5)]


@fixture()
def starcraft_preconditions_fluents_map() -> Dict[str, List[str]]:
    return json.load(open(STAR_CRAFT_FLUENTS_MAP_PATH, "rt"))


@fixture()
def starcraft_domain() -> Domain:
    return DomainParser(STAR_CRAFT_DOMAIN_PATH, partial_parsing=True).parse_domain()


@fixture()
def starcraft_observation(starcraft_domain: Domain) -> MultiAgentObservation:
    return TrajectoryParser(starcraft_domain).parse_trajectory(
        STAR_CRAFT_TRAJECTORY_PATH, executing_agents=STARCRAFT_AGENT_NAMES)


@fixture()
def starcraft_sam(starcraft_domain: Domain,
                  starcraft_preconditions_fluents_map: Dict[str, List[str]]) -> NumericMultiAgentSAM:
    return NumericMultiAgentSAM(
        starcraft_domain,
        polynomial_degree=0)


def test_learn_ma_domain_with_numeric_actions_works_on_starcraft_domain(
        starcraft_sam: NumericMultiAgentSAM, starcraft_observation: MultiAgentObservation):
    learned_model, _ = starcraft_sam.learn_action_model([starcraft_observation])
    print(learned_model.to_pddl())
