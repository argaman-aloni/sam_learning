import json
import pathlib
import signal
import sys
import time

from experiments.cluster_scripts.common import submit_job, progress_bar, sigint_handler


def execute_experiment_setup_batch(code_directory, configuration, environment_variables, experiment, experiment_index, total_run_time):
    print(f"Working on the experiment with domain {experiment['domain_file_name']}\n")
    fold_creation_sid = submit_job(
        conda_env="online_nsam",
        mem="6G",
        python_file=f"{code_directory}/folder_creation_for_parallel_execution.py",
        jobname=f"create_folds_job_{experiment['domain_file_name']}",
        suppress_output=False,
        arguments=[
            f"--working_directory_path {experiment['working_directory_path']}",
            f"--domain_file_name {experiment['domain_file_name']}",
            f"--learning_algorithms {','.join([str(e) for e in experiment['compared_versions']])}",
        ],
        environment_variables=environment_variables,
    )
    print(f"Submitted job with sid {fold_creation_sid}\n")
    progress_bar(experiment_index * configuration["num_folds"] + 1, total_run_time)
    time.sleep(1)
    print("Removing the temp.sbatch file")
    pathlib.Path("temp.sbatch").unlink()
    return fold_creation_sid


def main():
    signal.signal(signal.SIGINT, sigint_handler)
    print("Reading the configuration file.")
    configuration_file_path = sys.argv[1]
    with open(configuration_file_path, "rt") as configuration_file:
        configuration = json.load(configuration_file)

    environment_variables_path = configuration["environment_variables_file_path"]
    code_directory = configuration["code_directory_path"]
    workdir_path = pathlib.Path(configuration["working_directory_path"])
    with open(environment_variables_path, "rt") as environment_variables_file:
        environment_variables = json.load(environment_variables_file)

    arguments = ["driverlogHardNumeric.pddl", workdir_path]
    sid = submit_job(
        conda_env="online_nsam",
        mem="1G",
        python_file=f"{code_directory}/experiments_trajectories_creator.py",
        jobname=f"trajectory_creation",
        suppress_output=False,
        arguments=arguments,
        environment_variables=environment_variables,
    )
    print(f"Submitted job with sid {sid}\n")
    time.sleep(1)


if __name__ == "__main__":
    main()
