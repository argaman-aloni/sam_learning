"""Runs the experiments for the minecraft domain."""
import csv
import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Union

from pddl_plus_parser.lisp_parsers import TrajectoryParser, DomainParser

from sam_learning.core import LearnerDomain
from sam_learning.learners import NumericSAMLearner


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


def run_experiments(workdir_path: Path) -> None:
    """Runs the experiments for the starcrft domain.

    :param workdir_path: the directory containing the trajectories from the minecraft experiments.
    """
    minecraft_domain = DomainParser(workdir_path / "domain.pddl", partial_parsing=True).parse_domain()
    observations = []
    model_stats = []
    fluents_map = json.load(open(workdir_path / "fluents_map.json", "rt"))
    for trajectory_path in workdir_path.glob("*.trajectory"):
        print(f"Processing trajectory {trajectory_path}")
        observations.append(TrajectoryParser(minecraft_domain).parse_trajectory(trajectory_path))
        minecraft_sam = NumericSAMLearner(minecraft_domain, preconditions_fluent_map=fluents_map)
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
    run_experiments(Path("/home/mordocha/numeric_planning/domains/minecraft/"))
