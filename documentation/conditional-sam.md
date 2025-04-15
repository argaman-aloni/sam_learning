# Conditional SAM Documentation:

This contains the relevant information for the algorithm that supports learning action models with conditional effects.

## Supported Features
* Learning action models with conditional and universal effects.
* **Next**: Learning disjunctive preconditions.

## Usage

* First, you need to create your dataset. 
    * This stage requires having a problem generator at hand that can generate the problems you want to solve.
    * The generator should be able to generate the problems in the format of the PDDL files.
    * After creating the problems dataset, you should create the trajectories and the solutions for each problem.
    * This can be done using the \`[experiments_dataset_generator.py](../experiments/experiments_dataset_generator.py)` script.
    * run the script with the following command:
    ```python experiments/experiments_dataset_generator.py --help``` to see the options and the required command line arguments.
    * Then, run [conditional_effects_experiment_runner.py](../experiments/conditional_effects_experiment_runner.py) to run the code responsible for learning the action models with conditional effects.