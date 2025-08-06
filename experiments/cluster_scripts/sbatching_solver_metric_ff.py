import json
import pathlib
import signal
import sys
import time

from experiments.cluster_scripts.common import submit_job, sigint_handler


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
            configuration["tolerance"],
        ]
        print(f"Submitting job for problem {problem_file_path.stem}\n")
        print(f"Arguments: {arguments}\n")
        sid = submit_job(
            conda_env="online_nsam",
            mem="16G",
            python_file=f"{code_directory}/metric_ff_solver.py",
            jobname=f"metric_ff_solver",
            suppress_output=False,
            arguments=arguments,
            environment_variables=environment_variables,
        )
        print(f"Submitted job with sid {sid}\n")
        time.sleep(1)


if __name__ == "__main__":
    main()
