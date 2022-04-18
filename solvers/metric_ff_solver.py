import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import NoReturn

script_headers = """#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition main			### specify partition name where to run a job. short: 7 days limit; gtx1080: 7 days; debug: 2 hours limit and 1 job at a time
#SBATCH --time 0-03:30:00			### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name downward_planner_job			### name of the job
#SBATCH --output job-%J.out			### output log for running job - %J for job number
##SBATCH --mail-user=aaa.bbb@ccc	### user's email for sending job status messages
##SBATCH --mail-type=END			### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE

##SBATCH --mem=32G				### amount of RAM memory
##SBATCH --cpus-per-task=6			### number of CPU cores

module load anaconda
conda info --envs
source activate safe_action_learner
"""

script_execution_headers = """

directory_path={test_set_directory_path}
files_regex_path="$directory_path/{problems_regex}"
fast_downward_dir_path="/home/mordocha/downward"
fast_downward_file_path="$fast_downward_dir_path/fast-downward.py"
domain_path={domain_file_path}

"""

script_execution_lines = """

for FILE in $files_regex_path; do
  filename=$(basename -- "$FILE")
  name="${filename%.*}"
  echo "Processing the file - ${filename}:"
  if [[ $name == combined_domain ]]; then
    continue
  fi
  if test -f "${directory_path}/${name}_plan.solution"; then
    continue
  else
    python "$fast_downward_file_path" --overall-time-limit "1m" --sas-file "${directory_path}/${name}-output.sas" --plan-file "${directory_path}/${name}_plan.solution" "$domain_path" "$FILE" --evaluator "hff=ff()" --evaluator "hcea=cea()" --search "lazy_greedy([hff, hcea], preferred=[hff, hcea])"
    sleep 1
    rm "${directory_path}/${name}-output.sas"
  fi
done

"""

BATCH_JOB_SUBMISSION_REGEX = re.compile(b"Submitted batch job (?P<batch_id>\d+)")


class MetricFFSolver:
	logger: logging.Logger

	def __init__(self):
		self.logger = logging.getLogger(__name__)

	def write_batch_and_execute_solver(
			self, output_file_path: Path, test_set_directory_path: Path, problems_regex: str,
			domain_file_path: str) -> NoReturn:
		"""

		:param output_file_path:
		:param test_set_directory_path:
		:param problems_regex:
		:param domain_file_path:
		:return:
		"""
		os.chdir(output_file_path.parent)
		self.logger.info("Starting to solve the input problems using fast downward solver.")
		completed_file_str = script_headers
		completed_file_str += script_execution_headers.format(
			test_set_directory_path=str(test_set_directory_path),
			problems_regex=problems_regex,
			domain_file_path=domain_file_path
		)
		completed_file_str += script_execution_lines

		with open(output_file_path, "wt") as run_script_file:
			run_script_file.write(completed_file_str)

		submission_str = subprocess.check_output(["sbatch", str(output_file_path)])
		match = BATCH_JOB_SUBMISSION_REGEX.match(submission_str)
		batch_id = match.group("batch_id")
		# waiting for fast downward process to start
		time.sleep(1)
		execution_state = subprocess.check_output(["squeue", "--me"])
		while batch_id in execution_state:
			self.logger.debug(f"Solver with the id - {batch_id} is still running...")
			execution_state = subprocess.check_output(["squeue", "--me"])
			time.sleep(1)
			continue

		self.logger.info("Solver finished its execution!")

		self.logger.debug("Cleaning the sbatch file from the problems directory.")
		os.remove(output_file_path)
		return
