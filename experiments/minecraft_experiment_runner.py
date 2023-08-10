"""Runs the experiments for the minecraft domain."""
import csv
import datetime
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Union

from pddl_plus_parser.lisp_parsers import TrajectoryParser, DomainParser, ProblemParser
from pddl_plus_parser.models import GroundedPredicate

from sam_learning.core import LearnerDomain
from sam_learning.learners import NumericSAMLearner
from validators import run_validate_script


def export_learned_domain(workdir_path: Path, learned_domain: LearnerDomain, num_observations: int) -> Path:
    """Exports the learned domain into a file so that it will be used to solve the test set problems.

    :param workdir_path: the directory containing the trajectories from the minecraft experiments.
    :param learned_domain: the domain that was learned by the action model learning algorithm.
    """
    domain_file_name = learned_domain.name + f"_{num_observations}_trajectories_{datetime.date.today()}.pddl"
    domain_path = workdir_path / domain_file_name
    with open(domain_path, "wt") as domain_file:
        domain_file.write(learned_domain.to_pddl())

    return domain_path


def output_results(model_stats: List[Dict[str, Union[int, str]]], workdir_path: Path) -> None:
    """Outputs the results of the experiments to a csv file.

    :param model_stats: the statistics of the experiments.
    :param workdir_path: the directory containing the trajectories from the starcraft experiments.
    """
    with open(workdir_path / "statistics.csv", "wt", newline='') as statistics_file:
        stats_writer = csv.DictWriter(statistics_file, fieldnames=["num_trajectories", "num_actions", "runtime"])
        stats_writer.writeheader()
        for data_line in model_stats:
            stats_writer.writerow(data_line)


def add_noise_actions(agent_position, min_craft_sticks, plan, tree_to_place_tree_tap, unused_trees):
    if len(unused_trees) == 0:
        return agent_position, tree_to_place_tree_tap

    noise_actions = random.sample(unused_trees, k=random.randint(0, len(unused_trees) - 1))
    final_tree_for_goal = random.choice([tree_cell for tree_cell in unused_trees if tree_cell not in noise_actions
                                         and tree_cell != tree_to_place_tree_tap and tree_cell not in min_craft_sticks])
    for tree_cell in noise_actions:
        plan.append(f"(tp_to {agent_position.grounded_objects[0]} {tree_cell.grounded_objects[0]})")
        agent_position = GroundedPredicate(name="position", signature=agent_position.signature,
                                           object_mapping={"?c": tree_cell.grounded_objects[0]})
        plan.append(f"(break {tree_cell.grounded_objects[0]})")
        plan.append(f"(craft_plank)")
        plan.append(f"(craft_stick)")

    return agent_position, final_tree_for_goal


def create_expert_solution_files(workdir_path: Path, domain_file_name: str, problem_file_prefix: str) -> None:
    """

    :param workdir_path:
    :param domain_file_name:
    :param problem_file_prefix:
    :return:
    """
    minecraft_domain = DomainParser(workdir_path / domain_file_name, partial_parsing=False).parse_domain()
    for problem_path in workdir_path.glob(f"{problem_file_prefix}*.pddl"):
        problem_file_name_no_suffix = problem_path.stem
        if (workdir_path / f"{problem_file_name_no_suffix}.solution").exists():
            continue

        plan = []
        print(f"Processing problem {problem_path}")
        problem = ProblemParser(problem_path, minecraft_domain).parse_problem()
        initial_state_predicates = problem.initial_state_predicates
        tree_cells = [predicate for predicate in initial_state_predicates["(tree_cell ?c)"]]
        min_trees_to_success = random.sample(tree_cells, k=3)
        min_craft_sticks = random.sample(min_trees_to_success, k=2)
        agent_position = [predicate for predicate in initial_state_predicates["(position ?c)"]][0]
        for tree_cell in min_trees_to_success:
            cell = tree_cell.grounded_objects[0]
            plan.append(f"(tp_to {agent_position.grounded_objects[0]} {cell})")
            agent_position = GroundedPredicate(name="position", signature=agent_position.signature,
                                               object_mapping={"?c": cell})
            plan.append(f"(break {cell})")
            plan.append(f"(craft_plank)")
            if tree_cell in min_craft_sticks:
                plan.append(f"(craft_stick)")

        plan.append(f"(craft_tree_tap {min_trees_to_success[-1].grounded_objects[0]})")
        tree_to_place_tree_tap = random.choice(
            [tree_cell for tree_cell in tree_cells if tree_cell not in min_trees_to_success])
        plan.append(f"(tp_to crafting_table {tree_to_place_tree_tap.grounded_objects[0]})")
        agent_position = GroundedPredicate(name="position", signature=agent_position.signature,
                                           object_mapping={"?c": tree_to_place_tree_tap.grounded_objects[0]})
        plan.append(f"(place_tree_tap {tree_to_place_tree_tap.grounded_objects[0]})")
        unused_trees = [tree_cell for tree_cell in tree_cells if tree_cell not in min_trees_to_success and
                        tree_cell != tree_to_place_tree_tap]

        agent_position, final_tree_for_goal = add_noise_actions(agent_position, min_craft_sticks, plan,
                                                                tree_to_place_tree_tap, unused_trees)

        plan.append(f"(tp_to {agent_position.grounded_objects[0]} {final_tree_for_goal.grounded_objects[0]})")
        plan.append(f"(craft_wooden_pogo {final_tree_for_goal.grounded_objects[0]})")

        with open(workdir_path / f"{problem_file_name_no_suffix}.solution", "wt") as solution_file:
            solution_file.write("\n".join(plan))

        print(f"Solution for problem {problem_path} written to file")
        print("Validating solution....")
        validation_file_path = run_validate_script(
            domain_file_path=workdir_path / domain_file_name,
            problem_file_path=problem_path,
            solution_file_path=workdir_path / f"{problem_file_name_no_suffix}.solution")

        with open(validation_file_path, "r") as validation_file:
            validation_file_content = validation_file.read()
            assert "Plan valid" in validation_file_content


def run_experiments(workdir_path: Path) -> None:
    """Runs the experiments for the minecraft domain.

    :param workdir_path: the directory containing the trajectories from the minecraft experiments.
    """
    minecraft_domain = DomainParser(workdir_path / "domain.pddl", partial_parsing=True).parse_domain()
    observations = []
    model_stats = []
    for trajectory_path in workdir_path.glob("*.trajectory"):
        print(f"Processing trajectory {trajectory_path}")
        observations.append(TrajectoryParser(minecraft_domain).parse_trajectory(trajectory_path))
        minecraft_sam = NumericSAMLearner(minecraft_domain)
        learned_model, statistics = minecraft_sam.learn_action_model(observations)
        export_learned_domain(workdir_path, learned_model, len(observations))
        model_stats.append({"num_trajectories": len(observations),
                            "num_actions": len(learned_model.actions),
                            "runtime": statistics["learning_time"]})

    output_results(model_stats, workdir_path)


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG)
    create_expert_solution_files(workdir_path=Path("/home/mordocha/numeric_planning/domains/minecraft_real_map_small/"),
                                 domain_file_name="advanced_minecraft_domain.pddl",
                                 problem_file_prefix="advanced_map_instance")
