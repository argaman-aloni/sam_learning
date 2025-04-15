# Multi-Agent SAM Documentation:

This contains the relevant information for the multi-agent version of the SAM-Learning framework.

## Supported Features
* Learning action models when the agents are acting concurrently in the environment.
* Learning simple collaborative actions when the algorithm cannot differentiate between the agents' actions.
* Multi-Agent SAM+ supports learning macro actions in case the attributing the agents effects returns ambiguous results.
  * The macro actions enable monotonic learning since the provide mitigation in case the original algorithm cannot learn accurately.

## Installation

### Pre-requisites

The following are the pre-requisites for the SAM-Learning framework:
* Python 3.10 or higher.
* Installing all the dependencies using the requirements.txt file.
  * Most of the framework is dependent on the pddl-plus-parser package so it is important to install it.
  * The pddl-plus-parser package can be found [here](https://pypi.org/project/pddl-plus-parser/) and can be installed using `pip install pddl-plus-parser`.
* The code is dependent on the following external tools:
  * VAL - the plan validation algorithm.
  * Fast Downward planner.

Next we provide some information on the external tools and their installation process.

### Needed external tools:
3. **VAL plan validation algorithm** - can be downloaded from [here](https://github.com/KCL-Planning/VAL). Use the instructions given in the repository to install VAL.
4. **Fast Downward planner** - Discrete planner, used for the experiments. Can be downloaded from [here](https://www.fast-downward.org/HomePage).


### Environment variables

* 'METRIC_FF_DIRECTORY': set as empty string (external legacy dependency).
* 'ENHSP_FILE_PATH': set as empty string (external legacy dependency).
* 'VALIDATOR_DIRECTORY': Directory containing VAL compiled code.
  * \<path to containing directory>/VAL/
* 'FAST_DOWNWARD_DIR_PATH': The directory containing the compiled Fast Downward planner.
  * \<path to containing directory>/fast-downward-22.06/
* FF_DIRECTORY: The directory containing the FF planner.
  * \<path to containing directory>/FF-v2.3/

### Code initialization

* Clone the content of the repository from [GitHub](https://github.com/SPL-BGU/ma-sam.git).
* Install the dependencies using `pip install -r requirements.txt`.
* Set the required environment variables.
* Run the tests using `pytest` on the code to validate the installation.


## Dataset - were used in the experiments
The available dataset was used in the experiments that were conducted in the paper.
We experimented on the domains:
* Blocksworld
* Depots
* Driverlog
* Logistics
* Satellite
* Sokoban
* Rovers

The names of the agents for each domain:
* BlocksWorld - "[a1,a2,a3,a4]"
* Depots - "[depot0,depot1,depot2,depot3,distributor0,distributor1,distributor2,distributor3,driver0,driver1,driver2,driver3]"
* Driverlog - "[driver1,driver2,driver3,driver4,driver5,driver6,driver7,driver8]"
* Logistics - "[apn1,apn2,tru1,tru2,tru3,tru4,tru5]"
* Satellite - "[satellite0,satellite1,satellite2,satellite3,satellite4,satellite5,satellite6,satellite7,satellite8,satellite9]"
* Sokoban - "[player-01,player-02,player-03,player-04]"
* Rovers - "[rover0,rover1,rover2,rover3,rover4,rover5,rover6,rover7,rover8,rover9]"

## Usage
Note - the experiment framework is a bit deprecated and needs to be updated.

* The main algorithm is [multi_agent_sam.py](sam_learning/learners/multi_agent_sam.py). 
* The code contains the algorithm for learning multi-agent action models.
* The algorithm heavily relies on [sam_learning.py](sam_learning/learners/sam_learning.py) which is the lifted action model learning algorithm.
* There is some core functionality relevant to the CNF management is implemented in [literals_cnf.py](sam_learning/core/propositional_operations/literals_cnf.py).
* The extension of MA-SAM, i.e., MA-SAM+ is implemented in the module [multi_agent_sam_plus.py](sam_learning/learners/ma_sam_plus.py).

The experiments were run on a sbatch cluster using the concurrent_execution package.
specifically, the experiments presented in the paper were conducted using the [run_experiments.py](experiments/concurrent_execution/parallel_multi_agent_experiment_runner_with_triplets.py) script.
First you need to create the directories containing the experiment data using the [create_directories.py](experiments/concurrent_execution/folder_creation_for_parallel_execution.py) script.
* The required parameters for creating the directories for the experiments are:
  * --working_directory_path - the directory containing the domain's data - domain, problems and trajectories.
  * --domain_file_name - the domain file name.
    * E.g. satellite_combined_domain.pddl
  * --problem_prefix - the prefix of the problem files (used to iterate over files with specific names -- so that the code will not assume the domain file is a problem file).
  * --learning_algorithms - the learning algorithms to be used in the experiments these are enums representing the algorithms based on the definition in [util_types.py](utilities/util_types.py).
    * Specifically for these experiments use 1, 7, 25
  * --internal_iterations - the number of trajectories used as input per experiment iteration.
  * --experiment_size - the number of iterations to run for each experiment.
  * --no_random_trajectories - this is used for the performance evaluation. Just call this variable without any value.
* The code will create all the relevant folders and will copy the domain and problem files to the relevant directories.

After creating the directories, you can run the experiments using the [run_experiments.py](experiments/concurrent_execution/parallel_multi_agent_experiment_runner_with_triplets.py) script.
* The parameters for the execution of the experiments are:
  * --working_directory_path - same as before.
  * --domain_file_name - same as before.
  * --learning_algorithm - the algorithm currently being used (the number -- e.g. 7).
  * --fold_number - the number of the fold for the cross-validation (0-4). 
  * --executing_agents - the names of the agents in the domain.
    * E.g. "[satellite0,satellite1,satellite2,satellite3,satellite4,satellite5,satellite6,satellite7,satellite8,satellite9]"
  * --problem_prefix - the prefix of the problem files (used to iterate over files with specific names -- so that the code will not assume the domain file is a problem file).
  * --debug - a flag to run the code in debug mode -- will produce the log to STDOUT instead of only to a file. 
    * True / False

Example configuration file:
```json
{
  "environment_variables_file_path": "<path to the environment file in your computer>environment.json",
  "code_directory_path": "<path to the execution directory>/concurrent_execution",
  "experiments_script_path": "parallel_multi_agent_experiment_runner.py",
  "results_collection_script_name": "multi_agent_distributed_results_collector.py",
  "num_folds": 5,
  "experiment_configurations": [
	{
      "parallelization_data": {
        "min_index": 0,
        "max_index": 16,
        "hop": 1
      },
      "compared_versions": [
        1,
        7,
		25
      ],
      "working_directory_path": "experiments_dataset/blocks",
      "domain_file_name": "blocks_combined_domain.pddl",
      "problems_prefix": "probBLOCKS",
	  "executing_agents": "[a1,a2,a3,a4]"
    },
	{
      "parallelization_data": {
        "min_index": 0,
        "max_index": 12,
		"experiment_size": 15,
        "hop": 1
      },
      "compared_versions": [
        1,
        7,
		25
      ],
      "working_directory_path": "experiments_dataset/depots",
      "domain_file_name": "depot_combined_domain.pddl",
      "problems_prefix": "pfile",
	  "executing_agents": "[depot0,depot1,depot2,depot3,distributor0,distributor1,distributor2,distributor3,driver0,driver1,driver2,driver3]"
    },
	{
      "parallelization_data": {
        "min_index": 0,
        "max_index": 16,
        "hop": 1
      },
      "compared_versions": [
        1,
        7,
		25
      ],
      "working_directory_path": "experiments_dataset/driverlog",
      "domain_file_name": "driverlog_combined_domain.pddl",
      "problems_prefix": "pfile",
	  "executing_agents": "[driver1,driver2,driver3,driver4,driver5,driver6,driver7,driver8]"
    },
	{
      "parallelization_data": {
        "min_index": 0,
        "max_index": 16,
        "hop": 1
      },
      "compared_versions": [
        1,
        7,
		25
      ],
      "working_directory_path": "experiments_dataset/logistics",
      "domain_file_name": "logistics_combined_domain.pddl",
      "problems_prefix": "probLOGISTICS",
	  "executing_agents": "[apn1,apn2,tru1,tru2,tru3,tru4,tru5]"
    },
	{
      "parallelization_data": {
        "min_index": 0,
        "max_index": 16,
        "hop": 1
      },
      "compared_versions": [
        1,
        7,
		25
      ],
      "working_directory_path": "experiments_dataset/satellite",
      "domain_file_name": "satellite_combined_domain.pddl",
      "problems_prefix": "p",
	  "executing_agents": "[satellite0,satellite1,satellite2,satellite3,satellite4,satellite5,satellite6,satellite7,satellite8,satellite9]"
    },
	{
      "parallelization_data": {
        "min_index": 0,
        "max_index": 8,
		"experiment_size": 10,
        "hop": 1
      },
      "compared_versions": [
        1,
        7,
		25
      ],
      "working_directory_path": "experiments_dataset/sokoban",
      "domain_file_name": "sokoban-sequential_combined_domain.pddl",
      "problems_prefix": "p",
	  "executing_agents": "[player-01,player-02,player-03,player-04]"
    },
	{
      "parallelization_data": {
        "min_index": 0,
        "max_index": 80,
        "hop": 10
      },
      "compared_versions": [
        1,
        7,
		25
      ],
      "working_directory_path": "experiments_dataset/rovers",
      "domain_file_name": "rover_combined_domain.pddl",
      "problems_prefix": "pfile",
	  "executing_agents": "[rover0,rover1,rover2,rover3,rover4,rover5,rover6,rover7,rover8,rover9]"
    }
	]
}
```