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
    create_all_experiments_folders,
    submit_job_and_validate_execution,
    create_execution_arguments,
)

signal.signal(signal.SIGINT, sigint_handler)


def main():
    configurations = get_configurations()
    environment_variables = get_environment_variables(configurations)
    code_directory = configurations["code_directory_path"]
    print("Starting to setup and run the mult-agent experiments!")
    iterations_to_use = create_all_experiments_folders(code_directory, environment_variables, configurations, should_create_random_trajectories=False)
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
                            code_directory,
                            configurations,
                            experiment,
                            fold,
                            arguments,
                            environment_variables,
                            f"{experiment['domain_file_name']}_{fold}_multi_agent_experiment_runner",
                            None,
                        )

                    experiment_sids.append(sid)
                    formatted_date_time = datetime.now().strftime("%A, %B %d, %Y %I:%M %p")
                    print(
                        f"{formatted_date_time} - submitted job for the multi-agent experiments with sid {sid} for domain {experiment['domain_file_name']} fold {fold} and iteration {internal_iteration}."
                    )
                    pathlib.Path("temp.sbatch").unlink()
                    progress_bar(version_index, len(experiment["compared_versions"]))
                    arguments.pop(-1)  # removing the internal iteration from the arguments list

                print("Creating the job to run the experiment with triplets instead of trajectories.")
                submit_job_and_validate_execution(
                    code_directory,
                    configurations,
                    experiment,
                    fold,
                    arguments,
                    environment_variables,
                    f"triplets_{experiment['domain_file_name']}_{fold}_multi_agent_experiment_runner",
                    None,
                    python_file=f"{code_directory}/parallel_multi_agent_experiment_runner_with_triplets.py",
                )

            time.sleep(5)

        print("Finished building the experiment folds!")


if __name__ == "__main__":
    main()
