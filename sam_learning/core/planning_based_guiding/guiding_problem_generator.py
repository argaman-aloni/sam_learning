from pathlib import Path
from typing import List, Dict

from pddl_plus_parser.models import State, PDDLObject
from pddl_plus_parser.problem_generators import get_problem_template

TEMPLATE_FILE_PATH = Path("guiding_problem_template.pddl")


def generate_instance(domain_name: str, problem_object: Dict[str, PDDLObject],
                      current_state: State, goal_predicates: List[str]) -> str:
    """Generate a single planning problem instance.

    :param domain_name: the name of the domain.
    :param problem_object: the problem objects.
    :param current_state: the current state.
    :param goal_predicates: the goal predicates.
    """
    template = get_problem_template(TEMPLATE_FILE_PATH)
    template_mapping = {
        "instance_name": "planning_guide_problem_instance",
        "domain_name": domain_name,
        "objects": "\n".join([f"\t\t{str(obj)}" for obj in problem_object.values()]),
        "init": current_state.serialize(),
        "goal": "\n".join([f"\t\t\t{goal}" for goal in goal_predicates]),
    }
    return template.substitute(template_mapping)
