import pathlib
import signal
import string
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
    with open("experiments/cluster_scripts/template.sbatch", "rt", newline="\n") as template_file:
        text = template_file.read()
        sbatch_template = string.Template(text)

    # Complete the template code with the correct values
    template_mapping = {
        "job_name": jobname if jobname else "job",
        "cpus_per_task": cpus_per_task if cpus_per_task else 1,
        "dependency_exists": "#" if not dependency else "",
        "dependency": dependency or "",
        "mem": mem or "8G",
        "conda_env": conda_env or "online_nsam",
        "script": python_file,
        "arguments": " ".join(arguments) if arguments else "",
        "environment_variables": "\n".join([f"export {env_variable}={value}" for env_variable, value in environment_variables.items()]) if environment_variables else "",
    }

    sbatch_code = sbatch_template.substitute(template_mapping)

    with open("template.sbatch", "w+", newline="\n") as output_file:
        output_file.write(sbatch_code)

    data = subprocess.check_output(["sbatch", "template.sbatch", "--parsable"]).decode()
    return int(data.split()[-1])
