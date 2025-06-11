import pathlib
import signal
import time
from datetime import datetime

from experiments.cluster_scripts.common import (
    progress_bar,
    sigint_handler,
    get_configurations,
    get_environment_variables,
    EXPERIMENTS_CONFIG_STR,
    submit_job_and_validate_execution,
    create_execution_arguments,
    setup_experiments_folds_job,
    validate_job_running,
)

signal.signal(signal.SIGINT, sigint_handler)


def create_experiment_folders(
    code_directory,
    environment_variables,
    experiment,
):
    print(f"Creating the directories containing the folds datasets for the experiments.")
    sid = setup_experiments_folds_job(
        code_directory=code_directory,
        environment_variables=environment_variables,
        experiment=experiment,
        internal_iterations=None,
        experiment_size=100,
        should_create_random_trajectories=True,
    )
    return sid


def create_all_experiments_folders(
    code_directory,
    environment_variables,
    configurations,
):
    print("Creating the directories containing the folds datasets for the experiments.")
    jobs_sids = []
    output_internal_iterations = []
    for experiment_index, experiment in enumerate(configurations[EXPERIMENTS_CONFIG_STR]):
        fold_creation_sid = create_experiment_folders(
            code_directory,
            environment_variables,
            experiment,
        )
        jobs_sids.append(fold_creation_sid)
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
    create_all_experiments_folders(code_directory, environment_variables, configurations)
    for experiment_index, experiment in enumerate(configurations[EXPERIMENTS_CONFIG_STR]):
        experiment_sids = []
        for fold in range(configurations["num_folds"]):
            print(f"Working on fold {fold} of the experiment with domain {experiment['domain_file_name']}\n")
            for version_index, compared_version in enumerate(experiment["compared_versions"]):
                progress_bar(version_index, len(experiment["compared_versions"]))
                arguments = create_execution_arguments(experiment, fold, compared_version)
                sid = None
                while sid is None:
                    sid = submit_job_and_validate_execution(
                        code_directory,
                        configurations,
                        experiment,
                        fold,
                        arguments,
                        environment_variables,
                        f"{experiment['domain_file_name']}_{fold}_semi_online_experiment_runner",
                        None,
                    )

                experiment_sids.append(sid)
                formatted_date_time = datetime.now().strftime("%A, %B %d, %Y %I:%M %p")
                print(f"{formatted_date_time} - submitted job with sid {sid} for algorithm semi_online fold {fold}.")
                pathlib.Path("temp.sbatch").unlink()
                progress_bar(version_index, len(experiment["compared_versions"]))
                arguments.pop(-1)  # removing the internal iteration from the arguments list

            time.sleep(5)

        print("Finished building the experiment folds!")


if __name__ == "__main__":
    main()
