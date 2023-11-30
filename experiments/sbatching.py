import json
import os
import pathlib
import signal
import subprocess
import sys
import time


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
    with open("/home/mordocha/numeric_planning/domains/temp.sh", 'w+', newline="\n") as f:
        f.write('#!/bin/bash\n')
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
    total_run_time = configuration["num_folds"] * num_experiments + num_experiments
    progress_bar(0, total_run_time)
    for experiment_index, experiment in enumerate(configuration["experiment_configurations"]):
        fold_creation_sid = submit_job(
            conda_env='online_nsam', mem="32G",
            python_file=f"{code_directory}/folder_creation_for_parallel_execution.py",
            jobname=f"create_folds_job_{experiment['domain_file_name']}",
            suppress_output=False,
            arguments=[
                f"--working_directory_path {experiment['working_directory_path']}",
                f"--domain_file_name {experiment['domain_file_name']}",
            ],
            environment_variables=environment_variables)
        print(f"Submitted job with sid {fold_creation_sid}")
        progress_bar(experiment_index * configuration["num_folds"] + 1, total_run_time)
        pathlib.Path('/home/mordocha/numeric_planning/domains/temp.sh').unlink()
        print("Waiting for the fold creation job to finish")
        time.sleep(100)

        for fold in range(configuration["num_folds"]):
            current_iteration = (experiment_index + 1) * configuration["num_folds"] + fold + 2
            sid = submit_job(
                conda_env='online_nsam', mem="32G",
                python_file=f"{code_directory}/numeric_experiment_runner.py",
                jobname=f"run_experiment_{experiment['domain_file_name']}_{fold}",
                suppress_output=False,
                arguments=[
                    f"--working_directory_path {experiment['working_directory_path']}",
                    f"--domain_file_name {experiment['domain_file_name']}",
                    f"--solver_type {experiment['solver_type']}",
                    f"--learning_algorithm {experiment['learning_algorithm']}",
                    f"--fluents_map_path {experiment['fluents_map_path']}",
                    f"--problems_prefix {experiment['problems_prefix']}",
                    f"--num_folds {configuration['num_folds']}",
                    f"--fold_number {fold}"
                ],
                environment_variables=environment_variables)
            print(f"Submitted job with sid {sid}")
            progress_bar(current_iteration, total_run_time)
            time.sleep(5)


if __name__ == '__main__':
    main()
