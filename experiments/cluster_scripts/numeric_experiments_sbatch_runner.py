import json
import pathlib
import signal
import subprocess
import sys
import time
from datetime import datetime

from experiments.cluster_scripts.common import submit_job, progress_bar, sigint_handler

FIRST_BREAKPOINT = 10

EXPERIMENTS_CONFIG_STR = "experiment_configurations"

signal.signal(signal.SIGINT, sigint_handler)


def setup_experiments_folds_job(code_directory, environment_variables, experiment, internal_iterations, experiment_size):
    print(f"Working on the experiment with domain {experiment['domain_file_name']}\n")
    fold_creation_sid = submit_job(
        conda_env="online_nsam",
        mem="4G",
        python_file=f"{code_directory}/folder_creation_for_parallel_execution.py",
        jobname=f"create_folds_job_{experiment['domain_file_name']}",
        suppress_output=False,
        arguments=[
            f"--working_directory_path {experiment['working_directory_path']}",
            f"--domain_file_name {experiment['domain_file_name']}",
            f"--learning_algorithms {','.join([str(e) for e in experiment['compared_versions']])}",
            f"--internal_iterations {','.join([str(e) for e in internal_iterations])}",
            f"--problem_prefix {experiment['problems_prefix']}",
            f"--experiment_size {experiment_size}",
        ],
        environment_variables=environment_variables,
        logs_directory=pathlib.Path(experiment["working_directory_path"]) / "logs",
    )
    print(f"Submitted job with sid {fold_creation_sid}\n")
    time.sleep(1)
    print("Removing the temp.sbatch file")
    pathlib.Path("temp.sbatch").unlink()
    return fold_creation_sid


def execute_statistics_collection_job(code_directory, configuration, environment_variables, experiment, job_ids, internal_iterations):
    print(f"Creating the job that will collect the statistics from all the domain's experiments.")
    filtered_sids = [sid for sid in job_ids if validate_job_running(sid) is not None]
    statistics_collection_job = submit_job(
        conda_env="online_nsam",
        mem="4G",
        python_file=f"{code_directory}/distributed_results_collector.py",
        dependency=f"afterok:{':'.join([str(e) for e in filtered_sids])}",
        jobname=f"collect_statistics_{experiment['domain_file_name']}",
        suppress_output=False,
        arguments=[
            f"--working_directory_path {experiment['working_directory_path']}",
            f"--domain_file_name {experiment['domain_file_name']}",
            f"--learning_algorithms {','.join([str(e) for e in experiment['compared_versions']])}",
            f"--num_folds {configuration['num_folds']}",
            f"--internal_iterations {','.join([str(e) for e in internal_iterations])}",
        ],
        environment_variables=environment_variables,
    )
    print(f"Submitted job with sid {statistics_collection_job}\n")
    time.sleep(1)
    print("Removing the temp.sbatch for the statistics collection file")
    pathlib.Path("temp.sbatch").unlink()


def get_configurations():
    print("Reading the configuration file.")
    configuration_file_path = sys.argv[1]
    with open(configuration_file_path, "rt") as configuration_file:
        return json.load(configuration_file)


def get_environment_variables(configurations):
    environment_variables_path = configurations["environment_variables_file_path"]
    with open(environment_variables_path, "rt") as environment_variables_file:
        return json.load(environment_variables_file)


def create_execution_arguments(experiment, fold, compared_version):
    arguments = []
    arguments.append(f"--fold_number {fold}")
    arguments.append(f"--learning_algorithm {compared_version}")
    for key, value in experiment.items():
        if key != "compared_versions" and key != "parallelization_data":
            arguments.append(f"--{key} {value}")

    return arguments


def create_experiment_folders(code_directory, environment_variables, experiment):
    print(f"Creating the directories containing the folds datasets for the experiments.")
    parallelization_data = experiment["parallelization_data"]
    max_train_size = (
        int(parallelization_data["experiment_size"] * 0.8) + 1 if "experiment_size" in parallelization_data else parallelization_data["max_index"] + 1
    )
    if parallelization_data["hop"] == 100:
        internal_iterations = [1] + list(range(10, 100, 10)) + list(range(100, max_train_size, 100))
        print(f"Internal iterations: {internal_iterations}")
        sid = setup_experiments_folds_job(
            code_directory=code_directory,
            environment_variables=environment_variables,
            experiment=experiment,
            internal_iterations=internal_iterations,
            experiment_size=parallelization_data.get("experiment_size", 100),
        )
        return internal_iterations, sid

    internal_iterations = list(range(1, FIRST_BREAKPOINT)) + list(range(FIRST_BREAKPOINT, max_train_size, parallelization_data["hop"]))
    print(f"Internal iterations: {internal_iterations}")
    sid = setup_experiments_folds_job(
        code_directory=code_directory,
        environment_variables=environment_variables,
        experiment=experiment,
        internal_iterations=internal_iterations,
        experiment_size=parallelization_data.get("experiment_size", 100),
    )
    return internal_iterations, sid


def validate_job_running(sid):
    job_exists_command = ["squeue", "--job", f"{sid}"]
    try:
        result = subprocess.check_output(job_exists_command, shell=True).decode()
        if str(sid) in result:
            return sid

        return None

    except subprocess.CalledProcessError:
        return None


def submit_job_and_validate_execution(code_directory, configurations, experiment, fold, arguments, environment_variables, fold_creation_sid=None):
    job_name = f"{experiment['domain_file_name']}_{fold}_runner"
    dependency_argument = None if not fold_creation_sid else f"afterok:{fold_creation_sid}"
    sid = submit_job(
        conda_env="online_nsam",
        mem="16G",
        python_file=f"{code_directory}/{configurations['experiments_script_path']}",
        jobname=job_name,
        dependency=dependency_argument,
        suppress_output=False,
        arguments=arguments,
        environment_variables=environment_variables,
        logs_directory=pathlib.Path(experiment["working_directory_path"]) / "logs",
        suppress_error=True,
    )
    time.sleep(2)
    return validate_job_running(sid)


def create_all_experiments_folders(code_directory, environment_variables, configurations):
    print("Creating the directories containing the folds datasets for the experiments.")
    jobs_sids = []
    output_internal_iterations = []
    for experiment_index, experiment in enumerate(configurations[EXPERIMENTS_CONFIG_STR]):
        internal_iterations, fold_creation_sid = create_experiment_folders(code_directory, environment_variables, experiment)
        jobs_sids.append(fold_creation_sid)
        output_internal_iterations.append(internal_iterations)
        print(
            f"Submitted fold datasets folder creation job with the id {fold_creation_sid} for the experiment with domain {experiment['domain_file_name']}\n"
        )

    while any([validate_job_running(sid) is not None for sid in jobs_sids]):
        print("Waiting for the fold creation jobs to finish.")
        time.sleep(5)

    print("Finished building the experiment folds!")
    return output_internal_iterations


def main():
    configurations = get_configurations()
    environment_variables = get_environment_variables(configurations)
    code_directory = configurations["code_directory_path"]
    print("Starting to setup and run the experiments!")
    iterations_to_use = create_all_experiments_folders(code_directory, environment_variables, configurations)
    for experiment_index, experiment in enumerate(configurations[EXPERIMENTS_CONFIG_STR]):
        internal_iterations = iterations_to_use[experiment_index]
        experiment_sids = []
        for fold in range(configurations["num_folds"]):
            print(f"Working on fold {fold} of the experiment with domain {experiment['domain_file_name']}\n")
            for version_index, compared_version in enumerate(experiment["compared_versions"]):
                progress_bar(version_index, len(experiment["compared_versions"]))
                arguments = create_execution_arguments(experiment, fold, compared_version)
                for internal_iteration in internal_iterations:
                    arguments.append(f"--iteration_number {internal_iteration}")
                    sid = None
                    while sid is None:
                        sid = submit_job_and_validate_execution(
                            code_directory, configurations, experiment, fold, arguments, environment_variables, None
                        )

                    experiment_sids.append(sid)
                    formatted_date_time = datetime.now().strftime("%A, %B %d, %Y %I:%M %p")
                    print(f"{formatted_date_time} - submitted job with sid {sid} for fold {fold} and iteration {internal_iteration}.")
                    pathlib.Path("temp.sbatch").unlink()
                    progress_bar(version_index, len(experiment["compared_versions"]))
                    arguments.pop(-1)  # removing the internal iteration from the arguments list

            time.sleep(5)

        print("Finished building the experiment folds!")
        execute_statistics_collection_job(
            code_directory, configurations, environment_variables, experiment, experiment_sids, internal_iterations,
        )


if __name__ == "__main__":
    main()
