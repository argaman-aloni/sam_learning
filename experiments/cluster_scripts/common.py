import pathlib
import signal
import string
import subprocess
import time

JOB_ID_MESSAGE = 'echo -e "\\nSLURM_JOBID:\\t\\t" $SLURM_JOBID'
JOB_NODELIST_MESSAGE = 'echo -e "SLURM_JOB_NODELIST:\\t" $SLURM_JOB_NODELIST "\\n\\n"'


def sigint_handler(sig_num, frame):
    signal.signal(signal.SIGINT, sigint_handler)
    print("\nCtrl-C pressed. Do you want to quit? (y/n): ", end="")
    response = input().strip().lower()
    if response in ["y", "yes"]:
        print("Deleting temp.sbatch file.")
        pathlib.Path("temp.sbatch").unlink()
        exit(0)


def progress_bar(progress, total):
    percent = 100 * (progress / float(total))
    percent_to_show = int(percent) // 2
    bar = "❚" * percent_to_show + "-" * (50 - percent_to_show)
    print(f"\r|{bar}| {percent:.2f}%", end="\r")


def write_sbatch_and_submit_job(sbatch_code: str):
    with open("temp.sbatch", "w+", newline="\n") as output_file:
        output_file.write(sbatch_code)

    data = subprocess.check_output(["sbatch", "temp.sbatch", "--parsable"]).decode()
    time.sleep(1)
    return int(data.split()[-1])


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
    logs_directory=None,
):
    with open("experiments/cluster_scripts/template.sbatch", "rt", newline="\n") as template_file:
        text = template_file.read()
        sbatch_template = string.Template(text)

    # Complete the template code with the correct values
    all_environment_variables = [
        "export LOCAL_LOGS_PATH=/scratch/${SLURM_JOB_USER}/${SLURM_JOB_ID}",
        *[f"export {env_variable}={value}" for env_variable, value in environment_variables.items()],
    ]
    template_mapping = {
        "job_name": jobname if jobname else "job",
        "cpus_per_task": cpus_per_task if cpus_per_task else 1,
        "dependency_exists": "#" if not dependency else "",
        "dependency": dependency or "",
        "mem": mem or "8G",
        "conda_env": conda_env or "online_nsam",
        "script": python_file,
        "arguments": " ".join(arguments) if arguments else "",
        "environment_variables": "\n".join(all_environment_variables) if environment_variables else "",
        "job_info_print": JOB_ID_MESSAGE + "\n" + JOB_NODELIST_MESSAGE,
        "logs_dir": logs_directory or "/dev/null",
    }

    sbatch_code = sbatch_template.substitute(template_mapping)
    try:
        return write_sbatch_and_submit_job(sbatch_code)

    except subprocess.CalledProcessError as e:
        template_mapping["dependency_exists"] = "#"  # Remove the dependency if it exists
        template_mapping["dependency"] = ""  # Remove the dependency if it exists
        sbatch_code = sbatch_template.substitute(template_mapping)
        return write_sbatch_and_submit_job(sbatch_code)
