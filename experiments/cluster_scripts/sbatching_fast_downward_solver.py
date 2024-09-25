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
    print("Removing the temp.sh file")
    pathlib.Path("temp.sh").unlink()
    return fold_creation_sid


def execute_statistics_collection_job(code_directory, configuration, environment_variables, experiment, job_ids):
    print(f"Creating the job that will collect the statistics from all the domain's experiments.")
    statistics_collection_job = submit_job(
        conda_env="online_nsam",
        mem="6G",
        python_file=f"{code_directory}/distributed_results_collector.py",
        dependency=f"afterok:{':'.join([str(e) for e in job_ids])}",
        jobname=f"collect_statistics_{experiment['domain_file_name']}",
        suppress_output=False,
        arguments=[
            f"--working_directory_path {experiment['working_directory_path']}",
            f"--domain_file_name {experiment['domain_file_name']}",
            f"--learning_algorithms {','.join([str(e) for e in experiment['compared_versions']])}",
            f"--num_folds {configuration['num_folds']}",
        ],
        environment_variables=environment_variables,
    )
    print(f"Submitted job with sid {statistics_collection_job}\n")
    time.sleep(1)
    print("Removing the temp.sh for the statistics collection file")
    pathlib.Path("temp.sh").unlink()


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

    for problem_file_path in workdir_path.glob(f"{configuration['problems_prefix']}*.pddl"):
        arguments = [
            str((workdir_path / configuration["domain_file_name"]).absolute()),
            str(problem_file_path.absolute()),
            str(workdir_path),
            configuration["timeout"],
        ]
        print(f"Submitting job for problem {problem_file_path.stem}\n")
        sid = submit_job(
            conda_env="online_nsam",
            mem="16G",
            python_file=f"{code_directory}/fast_downward_solver.py",
            jobname=f"fast_downward_solver",
            suppress_output=False,
            arguments=arguments,
            environment_variables=environment_variables,
        )
        print(f"Submitted job with sid {sid}\n")
        time.sleep(1)


if __name__ == "__main__":
    main()
