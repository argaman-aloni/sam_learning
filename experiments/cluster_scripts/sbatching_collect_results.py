import json
import pathlib
import signal
import sys
import time

from experiments.cluster_scripts.common import submit_job, sigint_handler


def execute_statistics_collection_job(code_directory, configuration, environment_variables, experiment):
    print(f"Creating the job that will collect the statistics from all the domain's experiments.")
    parallelization_data = experiment["parallelization_data"]
    experiment_internal_iterations = list(range(parallelization_data["min_index"], parallelization_data["max_index"], parallelization_data["hop"]))
    statistics_collection_job = submit_job(
        conda_env="online_nsam",
        mem="6G",
        python_file=f"{code_directory}/distributed_results_collector.py",
        jobname=f"collect_statistics_{experiment['domain_file_name']}",
        suppress_output=False,
        arguments=[
            f"--working_directory_path {experiment['working_directory_path']}",
            f"--domain_file_name {experiment['domain_file_name']}",
            f"--learning_algorithms {','.join([str(e) for e in experiment['compared_versions']])}",
            f"--num_folds {configuration['num_folds']}",
            f"--internal_iterations {','.join([str(e) for e in experiment_internal_iterations])}",
        ],
        environment_variables=environment_variables,
    )
    print(f"Submitted job with sid {statistics_collection_job}\n")
    time.sleep(1)
    print("Removing the temp.sbatch for the statistics collection file")
    pathlib.Path("temp.sbatch").unlink()


def main():
    signal.signal(signal.SIGINT, sigint_handler)
    print("Reading the configuration file.")
    configuration_file_path = sys.argv[1]
    with open(configuration_file_path, "rt") as configuration_file:
        configuration = json.load(configuration_file)

    environment_variables_path = configuration["environment_variables_file_path"]
    code_directory = configuration["code_directory_path"]
    with open(environment_variables_path, "rt") as environment_variables_file:
        environment_variables = json.load(environment_variables_file)

    print("Submitted fold creation job")
    num_experiments = len(configuration["experiment_configurations"])
    total_run_time = configuration["num_folds"] * num_experiments * 4 + num_experiments
    progress_bar(0, total_run_time)
    for experiment_index, experiment in enumerate(configuration["experiment_configurations"]):
        execute_statistics_collection_job(code_directory, configuration, environment_variables, experiment)


if __name__ == "__main__":
    main()
