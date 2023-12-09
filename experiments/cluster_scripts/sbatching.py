import json
import pathlib
import signal
import subprocess
import sys
import time
from datetime import datetime


def sigint_handler(sig, frame):
    signal.signal(signal.SIGINT, sigint_handler)
    print('\nCtrl-C pressed. Do you want to quit? (y/n): ', end="")
    response = input().strip().lower()
    if response in ['y', 'yes']:
        pathlib.Path('temp.sh').unlink()
        exit(0)


def progress_bar(progress, total):
    percent = 100 * (progress / float(total))
    percent_to_show = int(percent) // 2
    bar = '❚' * percent_to_show + '-' * (50 - percent_to_show)
    print(f'\r|{bar}| {percent:.2f}%', end='\r')


def submit_job(
        dependency=None, mem=None, runtime=None, jobname=None, cpus_per_task=None, suppress_error=False,
        error_file=None, suppress_output=False, output_file=None, conda_env=None,
        python_file=None, arguments=None, environment_variables=None):
    with open("temp.sh", 'w+', newline="\n") as f:
        f.write('#!/bin/bash\n')

        f.write("#SBATCH --partition main\n")

        if runtime:
            f.write(f'#SBATCH --time {runtime}\n')
        else:
            f.write('#SBATCH --time 7-00:00:00\n')

        if jobname:
            f.write(f'#SBATCH --job-name {jobname}\n')

        if cpus_per_task:
            f.write(f'#SBATCH --cpus-per-task={cpus_per_task}\n')
        else:
            f.write('#SBATCH --cpus-per-task=1\n')

        if dependency:
            f.write(f'#SBATCH --dependency={dependency}')

        if mem:
            f.write(f'#SBATCH --mem={mem}')

        if suppress_output:
            f.write('#SBATCH --output /dev/null\n')
        else:
            if output_file:
                f.write(f'#SBATCH --output {output_file}\n')

        if suppress_error:
            f.write('#SBATCH --error /dev/null\n')
        else:
            if error_file:
                f.write(f'#SBATCH --error {error_file}\n')

        f.write('\n\n')

        f.write("echo `date`\n")
        f.write('echo -e "\\nSLURM_JOBID:\\t\\t" $SLURM_JOBID\n')
        f.write('echo -e "SLURM_JOB_NODELIST:\\t" $SLURM_JOB_NODELIST "\\n"\n')

        f.write('\n\n')

        # setting up environment variables
        if environment_variables:
            for env_variable, value in environment_variables.items():
                f.write(f'export {env_variable}={value}\n')

        f.write('\n\n')

        if conda_env:
            f.write('module load anaconda\n')
            f.write(f'source activate {conda_env}\n')

        f.write(f"python {python_file} {' '.join(arguments)}\n")

    data = subprocess.check_output(['sbatch', 'temp.sh', '--parsable']).decode()
    return int(data.split()[-1])


def execute_experiment_setup_batch(
        code_directory, configuration, environment_variables, experiment, experiment_index, total_run_time,
        internal_iterations):
    print(f"Working on the experiment with domain {experiment['domain_file_name']}\n")
    fold_creation_sid = submit_job(
        conda_env='online_nsam', mem="6G",
        python_file=f"{code_directory}/folder_creation_for_parallel_execution.py",
        jobname=f"create_folds_job_{experiment['domain_file_name']}",
        suppress_output=False,
        arguments=[
            f"--working_directory_path {experiment['working_directory_path']}",
            f"--domain_file_name {experiment['domain_file_name']}",
            f"--learning_algorithms {','.join([str(e) for e in experiment['compared_versions']])}",
            f"--internal_iterations {','.join([str(e) for e in internal_iterations])}"
        ],
        environment_variables=environment_variables)
    print(f"Submitted job with sid {fold_creation_sid}\n")
    progress_bar(experiment_index * configuration["num_folds"] + 1, total_run_time)
    time.sleep(1)
    print("Removing the temp.sh file")
    pathlib.Path('temp.sh').unlink()
    return fold_creation_sid


def execute_statistics_collection_job(code_directory, configuration, environment_variables, experiment, job_ids,
                                      internal_iterations):
    print(f"Creating the job that will collect the statistics from all the domain's experiments.")
    statistics_collection_job = submit_job(
        conda_env='online_nsam', mem="6G",
        python_file=f"{code_directory}/distributed_results_collector.py",
        dependency=f"afterok:{':'.join([str(e) for e in job_ids])}",
        jobname=f"collect_statistics_{experiment['domain_file_name']}",
        suppress_output=False,
        arguments=[
            f"--working_directory_path {experiment['working_directory_path']}",
            f"--domain_file_name {experiment['domain_file_name']}",
            f"--learning_algorithms {','.join([str(e) for e in experiment['compared_versions']])}",
            f"--num_folds {configuration['num_folds']}",
            f"--internal_iterations {','.join([str(e) for e in internal_iterations])}"
        ],
        environment_variables=environment_variables)
    print(f"Submitted job with sid {statistics_collection_job}\n")
    time.sleep(1)
    print("Removing the temp.sh for the statistics collection file")
    pathlib.Path('temp.sh').unlink()


def write_contextual_sids_to_file(contextual_sids):
    with open("contextual_sids.json", "wt") as contextual_sids_file:
        json.dump(contextual_sids, contextual_sids_file)


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
    experiment_termination_ids = {}
    for experiment_index, experiment in enumerate(configuration["experiment_configurations"]):
        parallelization_data = experiment["parallelization_data"]
        experiment_internal_iterations = list(
            range(parallelization_data["min_index"], parallelization_data["max_index"], parallelization_data["hop"]))
        fold_creation_sid = execute_experiment_setup_batch(
            code_directory, configuration, environment_variables,
            experiment, experiment_index, total_run_time, experiment_internal_iterations)

        experiment_termination_ids[f"{experiment['domain_file_name']}"] = []
        for fold in range(configuration["num_folds"]):
            for version_index, compared_version in enumerate(experiment["compared_versions"]):
                current_iteration = (experiment_index + 1) * configuration["num_folds"] * version_index + fold + 2
                arguments = [f"--{key} {value}" for key, value in experiment.items() if key != "compared_versions" and key != "parallelization_data"]
                arguments.append(f"--fold_number {fold}")
                arguments.append(f"--learning_algorithm {compared_version}")
                for internal_iteration in experiment_internal_iterations:
                    arguments.append(f"--iteration_number {internal_iteration}")
                    sid = submit_job(
                        conda_env='online_nsam', mem="64G",
                        python_file=f"{code_directory}/{configuration['experiments_script_path']}",
                        jobname=f"{experiment['domain_file_name']}_{fold}_{internal_iteration}_run_experiments",
                        dependency=f"afterok:{fold_creation_sid}",
                        suppress_output=False,
                        arguments=arguments,
                        environment_variables=environment_variables)
                    formatted_date_time = datetime.now().strftime("%A, %B %d, %Y %I:%M %p")
                    print(f"Current date and time: {formatted_date_time}")
                    print(f"{formatted_date_time} Submitted job with sid {sid}")
                    # maintaining the IDs of the experiment jobs so that once they are all done a job that
                    # collects the data will be called and will combine the data together.
                    experiment_termination_ids[f"{experiment['domain_file_name']}"].append(sid)
                    progress_bar(current_iteration, total_run_time)
                    time.sleep(5)
                    pathlib.Path('temp.sh').unlink()

                    arguments.pop(-1)   # removing the internal iteration from the arguments list

            time.sleep(600)

        print("Finished building the experiment folds!")
        execute_statistics_collection_job(
            code_directory, configuration, environment_variables,
            experiment, experiment_termination_ids[f"{experiment['domain_file_name']}"], experiment_internal_iterations)

    write_contextual_sids_to_file(experiment_termination_ids)


if __name__ == '__main__':
    main()
