import pathlib
import signal
import subprocess


def sigint_handler(sig_num, frame):
    signal.signal(signal.SIGINT, sigint_handler)
    print("\nCtrl-C pressed. Do you want to quit? (y/n): ", end="")
    response = input().strip().lower()
    if response in ["y", "yes"]:
        print("Deleting temp.sh file.")
        pathlib.Path("temp.sh").unlink()
        exit(0)


def progress_bar(progress, total):
    percent = 100 * (progress / float(total))
    percent_to_show = int(percent) // 2
    bar = "❚" * percent_to_show + "-" * (50 - percent_to_show)
    print(f"\r|{bar}| {percent:.2f}%", end="\r")


def submit_job(
    dependency=None,
    mem=None,
    runtime=None,
    jobname=None,
    cpus_per_task=None,
    suppress_error=False,
    error_file=None,
    suppress_output=False,
    output_file=None,
    conda_env=None,
    python_file=None,
    arguments=None,
    environment_variables=None,
):
    with open("temp.sh", "w+", newline="\n") as f:
        f.write("#!/bin/bash\n")

        f.write("#SBATCH --partition main\n")

        if runtime:
            f.write(f"#SBATCH --time {runtime}\n")
        else:
            f.write("#SBATCH --time 7-00:00:00\n")

        if jobname:
            f.write(f"#SBATCH --job-name {jobname}\n")

        if cpus_per_task:
            f.write(f"#SBATCH --cpus-per-task={cpus_per_task}\n")
        else:
            f.write("#SBATCH --cpus-per-task=1\n")

        if dependency:
            f.write(f"#SBATCH --dependency={dependency}")

        if mem:
            f.write(f"#SBATCH --mem={mem}")

        if suppress_output:
            f.write("#SBATCH --output /dev/null\n")
        else:
            if output_file:
                f.write(f"#SBATCH --output {output_file}\n")

        if suppress_error:
            f.write("#SBATCH --error /dev/null\n")
        else:
            if error_file:
                f.write(f"#SBATCH --error {error_file}\n")

        f.write("\n\n")

        f.write("echo `date`\n")
        f.write('echo -e "\\nSLURM_JOBID:\\t\\t" $SLURM_JOBID\n')
        f.write('echo -e "SLURM_JOB_NODELIST:\\t" $SLURM_JOB_NODELIST "\\n"\n')

        f.write("\n\n")

        # setting up environment variables
        if environment_variables:
            for env_variable, value in environment_variables.items():
                f.write(f"export {env_variable}={value}\n")

        f.write("\n\n")

        if conda_env:
            f.write("module load anaconda\n")
            f.write(f"source activate {conda_env}\n")

        f.write(f"python {python_file} {' '.join(arguments)}\n")

    data = subprocess.check_output(["sbatch", "temp.sh", "--parsable"]).decode()
    return int(data.split()[-1])
