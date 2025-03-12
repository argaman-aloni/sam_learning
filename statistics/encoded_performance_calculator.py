import uuid
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Callable, Union

from pddl_plus_parser.exporters import ProblemExporter
from pddl_plus_parser.models import (
    Domain,
    Observation,
    ActionCall,
    State,
    Operator,
    PDDLObject,
    Problem,
)

from semantic_performance_calculator import (SemanticPerformanceCalculator, _calculate_precision_recall)
from utilities import LearningAlgorithmType
from validators import run_validate_script, VALID_PLAN


class EncodedPerformanceCalculator(SemanticPerformanceCalculator):
    def __init__(
        self,
        model_domain: Domain,
        model_domain_path: Path,
        observations: List[Observation],
        working_directory_path: Path,
        learning_algorithm: LearningAlgorithmType,
        encoders: Dict[str, list[Callable[[ActionCall], ActionCall]]] = None,
        decoders: Dict[str, Callable[[ActionCall], ActionCall]] = None):
        super().__init__(model_domain, model_domain_path, observations, working_directory_path, learning_algorithm)

        self.encoders = encoders
        self.decoders = decoders


    @staticmethod
    def _calculate_applicability_in_state(problem_file_path: Path, solution_file_path: Path, domain_file_path: Path) -> bool:
        """Calculate whether the action is applicable in the state.

        :param problem_file_path: the path to the problem file.
        :param solution_file_path: the path to the solution file.
        :param domain_file_path: the path to the domain file.
        :return: whether the action is applicable in the state.
        """
        validation_file_path = run_validate_script(
            domain_file_path=domain_file_path, problem_file_path=problem_file_path, solution_file_path=solution_file_path
        )
        with open(validation_file_path, "r", encoding="utf-8") as validation_file:
            validation_file_content = validation_file.read()

        validation_file_path.unlink()
        return VALID_PLAN in validation_file_content

    def _calculate_action_applicability_rate(
        self, action_call: ActionCall, learned_domain_path: Path, observed_state: State, problem_objects: Dict[str, PDDLObject],
            possible_proxy_calls=None
    ) -> Tuple[int, int, int]:
        """Test whether an action is applicable in both the model domain and the generated domain.

        :param action_call: the action call that is tested for applicability.
        :param learned_domain_path: the domain that was learned using the action model learning algorithm.
        :param observed_state: the state that was observed in the trajectory data.
        :param problem_objects: the objects that were used in the problem definition.
        :param possible_proxy_calls: possible calls for action if injective binding assumption does not hold.
        :return: a tuple containing the number of true positives, false positives and false negatives.
        """
        if possible_proxy_calls is None:
            possible_proxy_calls = [action_call]

        self.logger.debug(f"Calculating the applicability rate for the action - {action_call.name}")
        applicability_validation_problem = Problem(domain=self.model_domain)
        applicability_validation_problem.name = f"instance_{uuid.uuid4()}"
        applicability_validation_problem.objects = problem_objects
        applicability_validation_problem.initial_state_predicates = observed_state.state_predicates
        applicability_validation_problem.initial_state_fluents = observed_state.state_fluents
        self.logger.debug(
            f"Exporting a problem with the initial state as in {str(observed_state)} and no goals to validate whether the action is applicable."
        )
        current_problem_file_path = self.temp_dir_path / f"applicability_validation_problem_{uuid.uuid4()}.pddl"
        current_solution_file_path = self.temp_dir_path / f"applicability_validation_solution_{uuid.uuid4()}.solution"
        current_encoded_solution_file_path = self.temp_dir_path / f"applicability_validation_encoded_solution_{uuid.uuid4()}.solution"

        with open(current_solution_file_path, "wt") as solution_file:
            solution_file.write(str(action_call))

        applicable_in_model = self._calculate_applicability_in_state(current_problem_file_path,
                                                                     current_solution_file_path,
                                                                     self.model_domain_path)
        ProblemExporter().export_problem(problem=applicability_validation_problem,
                                         export_path=current_problem_file_path)

        # unlink original action solution file
        current_solution_file_path.unlink()
        # iterate over possible proxy actions
        proxy_results: Dict[str, Tuple[int, int, int]] = {}
        for proxy_call in possible_proxy_calls:
            with open(current_encoded_solution_file_path, "wt") as encoded_solution_file:
                encoded_solution_file.write(str(proxy_call))
            self.logger.debug(
                f"Exported the problem to {current_problem_file_path}, now validating the action's applicability.")

            applicable_in_learned = self._calculate_applicability_in_state(current_problem_file_path,
                                                                           current_encoded_solution_file_path,
                                                                           learned_domain_path)
            # unlink proxy action solution file
            current_encoded_solution_file_path.unlink()

            proxy_results[proxy_call.name] = \
                (
                    int(applicable_in_learned == applicable_in_model and applicable_in_learned),
                    int(applicable_in_learned and not applicable_in_model),
                    int(not applicable_in_learned and applicable_in_model),
                )

            # check if FP. FP-> unsafe so, raise warning and record unsafe action
            if applicable_in_learned and not applicable_in_model:
                self.logger.warning(
                    f"action {action_call.name} is not applicable in model domain "
                    f"yet proxy {proxy_call.name}, is applicable in learned model, check safety of learning!")
                current_problem_file_path.unlink()
                return proxy_results[proxy_call.name]

            if applicable_in_learned and applicable_in_model:
                current_problem_file_path.unlink()
                return proxy_results[proxy_call.name]

        current_problem_file_path.unlink()
        return proxy_results[possible_proxy_calls[0].name]


    def _calculate_effects_difference_rate(
        self,
        observation: Observation,
        learned_domain: Domain,
        num_false_negatives: Dict[str, int],
        num_false_positives: Dict[str, int],
        num_true_positives: Dict[str, int],
    ) -> None:
        """Calculates the effects difference rate for each action.

        :param observation: the observation that is being tested.
        :param learned_domain: the domain that was learned using the action model learning algorithm.
        :param num_false_negatives: the dictionary mapping between the action name and the number of false negative
        :param num_false_positives: the dictionary mapping between the action name and the number of false positive
        :param num_true_positives: the dictionary mapping between the action name and the number of true positive
        """
        self.logger.info("Calculating effects difference rate for the observation.")

        for observation_triplet in observation.components:
            model_previous_state = observation_triplet.previous_state
            executed_action = observation_triplet.grounded_action_call
            model_next_state = observation_triplet.next_state
            possible_proxy_calls: List[ActionCall] = []
            if executed_action.name not in learned_domain.actions:
                if executed_action not in self.encoders:
                    possible_proxy_calls = [executed_action]
                else:
                    possible_proxy_calls = [encode(executed_action) for encode in self.encoders[executed_action.name]]

            proxy_operators: Dict[str, Union[Operator,None]] = {}
            for proxy_call in possible_proxy_calls:
                try:
                    learned_operator = Operator(
                        action=learned_domain.actions[proxy_call.name],
                        domain=learned_domain,
                        grounded_action_call=proxy_call.parameters,
                        problem_objects=observation.grounded_objects,
                    )
                    proxy_operators[proxy_call.name] = learned_operator

                except ValueError:
                    proxy_operators[proxy_call.name] = None

            if all(proxy_operator is None for proxy_operator in proxy_operators.values()):
                self.logger.debug("The action is not applicable in the state.")
                continue

            else:
                for proxy_name, proxy_operator in proxy_operators.items():
                    if proxy_operator is not None:
                        try:
                            learned_next_state = proxy_operator.apply(model_previous_state)
                            self.logger.debug("Validating if there are any false negatives.")
                            model_state_predicates = {
                                predicate.untyped_representation
                                for grounded_predicate in model_next_state.state_predicates.values()
                                for predicate in grounded_predicate
                            }

                            learned_state_predicates = {
                                predicate.untyped_representation
                                for grounded_predicate in learned_next_state.state_predicates.values()
                                for predicate in grounded_predicate
                            }

                            if len(model_state_predicates) == 0 and len(learned_state_predicates) == 0:
                                num_true_positives[executed_action.name] += 1
                                break

                            num_true_positives[executed_action.name] += len(
                                model_state_predicates.intersection(learned_state_predicates))
                            num_false_positives[executed_action.name] += len(
                                learned_state_predicates.difference(model_state_predicates))
                            num_false_negatives[executed_action.name] += len(
                                model_state_predicates.difference(learned_state_predicates))
                            break

                        except ValueError:
                            # not applicable, try another proxy
                            continue


    def calculate_preconditions_semantic_performance(
        self, learned_domain: Domain, learned_domain_path: Path
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculates the precision recall values of the learned preconditions.

        :param learned_domain: the action model that was learned using the action model learning algorithm
        :param learned_domain_path: the path to the learned domain.
        :return: the precision and recall dictionaries.
        """
        num_true_positives = defaultdict(int)
        num_false_negatives = defaultdict(int)
        num_false_positives = defaultdict(int)
        self.logger.debug("Starting to calculate the semantic preconditions performance")
        for index, observation in enumerate(self.dataset_observations):
            observation_objects = observation.grounded_objects
            for component in observation.components:
                action = component.grounded_action_call
                if action.name not in learned_domain.actions and action.name not in self.encoders:
                    continue

                possible_proxy_calls = None
                if action.name in self.encoders:
                    possible_proxy_calls = [encode(action) for encode in self.encoders[action.name]]

                true_positive, false_positive, false_negative = self._calculate_action_applicability_rate(
                    action, learned_domain_path, component.previous_state, observation_objects, possible_proxy_calls
                )

                num_true_positives[action.name] += true_positive
                num_false_positives[action.name] += false_positive
                num_false_negatives[action.name] += false_negative

        return _calculate_precision_recall(num_false_negatives, num_false_positives, num_true_positives, list(learned_domain.actions.keys()))
