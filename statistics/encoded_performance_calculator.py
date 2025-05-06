from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Callable, Set

from pddl_plus_parser.models import (
    Domain,
    Observation,
    ActionCall,
    State,
    Operator,
    PDDLObject,
)

from statistics.semantic_performance_calculator import SemanticPerformanceCalculator, _calculate_precision_recall
from utilities import LearningAlgorithmType


class EncodedPerformanceCalculator(SemanticPerformanceCalculator):
    def __init__(
            self,
            model_domain: Domain,
            model_domain_path: Path,
            observations: List[Observation],
            working_directory_path: Path,
            learning_algorithm: LearningAlgorithmType,
            encode: Callable[[ActionCall], List[ActionCall]] = None,
            decode: Callable[[ActionCall], ActionCall] = None
    ):
        super().__init__(model_domain, model_domain_path, observations, working_directory_path, learning_algorithm)
        self.encode = encode
        self.decode = decode


    def _calculate_action_applicability_rate(
            self, action_call: ActionCall, learned_domain_path: Path, observed_state: State,
            problem_objects: Dict[str, PDDLObject], learned_domain: Domain=None
    ) -> Tuple[int, int, int]:
        """Test whether an action is applicable in both the model domain and the generated domain.

        :param action_call: the action call that is tested for applicability.
        :param learned_domain_path: the domain that was learned using the action model learning algorithm.
        :param observed_state: the state that was observed in the trajectory data.
        :param problem_objects: the objects that were used in the problem definition.
        :return: a tuple containing the number of true positives, false positives and false negatives.
        """

        self.logger.debug(f"Calculating the applicability rate for the action - {action_call.name}")

        # check applicability
        applicable_in_model = self.is_applicable(action_call=action_call,
                                                 observed_state=observed_state,
                                                 problem_objects=problem_objects)

        applicable_in_learned = False
        try:
            learned_action_calls = self.encode(action_call)
            if not learned_action_calls:
                learned_action_calls = []
        except Exception:
            learned_action_calls = []

        for learned_call in learned_action_calls:
            #check proxy call applicability
            if learned_call.name in learned_domain.actions:
                applicable_in_learned = self.is_applicable(action_call=learned_call,
                                                           observed_state=observed_state,
                                                           problem_objects=problem_objects,
                                                           learned_domain_path=learned_domain_path)
            if applicable_in_learned:
                if not applicable_in_model:  # alert unsafe action existence
                    self.logger.warning(
                        f"action {action_call.name} is not applicable in model domain "
                        f"yet the learned action: {learned_call.name},"
                        f" is applicable in learned model, check safety of learning!")
                break  # meaning one proxy action allows applying the action call no need for further testing

        res = (
            int(applicable_in_learned == applicable_in_model and applicable_in_learned),
            int(applicable_in_learned and not applicable_in_model),
            int(not applicable_in_learned and applicable_in_model),)

        return res

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
        print("encoded")

        for observation_triplet in observation.components:
            model_previous_state = observation_triplet.previous_state
            executed_action = observation_triplet.grounded_action_call
            model_next_state = observation_triplet.next_state

            prev_state_predicates = {predicate.untyped_representation
                                     for grounded_predicate in model_previous_state.state_predicates.values()
                                     for predicate in grounded_predicate}

            next_state_predicates = {predicate.untyped_representation
                                     for grounded_predicate in model_next_state.state_predicates.values()
                                     for predicate in grounded_predicate}

            original_effects: Set[str] = next_state_predicates.difference(prev_state_predicates)

            try:
                possible_learned_calls = self.encode(executed_action)
                if not possible_learned_calls:
                    possible_learned_calls = []
            except Exception:
                possible_learned_calls = []

            if len(possible_learned_calls) == 0 or not possible_learned_calls:
                # if action was not learned, treat learned model as it learned the action but has no effects
                num_false_negatives[executed_action.name] += len(original_effects)
                continue

            learned_effects: Set[str] = set()
            for learned_call in possible_learned_calls:
                try:
                    learned_operator = Operator(action=learned_domain.actions[learned_call.name],
                                                domain=learned_domain,
                                                grounded_action_call=learned_call.parameters,
                                                problem_objects=observation.grounded_objects)

                    if learned_operator.is_applicable(model_previous_state):
                        learned_next_state = learned_operator.apply(model_previous_state)
                        learned_state_predicates = {predicate.untyped_representation
                                                    for grounded_predicate in
                                                    learned_next_state.state_predicates.values()
                                                    for predicate in grounded_predicate}
                        learned_effects = learned_state_predicates.difference(prev_state_predicates)
                        break

                except ValueError:
                    continue
                except KeyError:
                    continue

            # original effects = next_state_predicates - prev_state_predicates
            # learned domain effects fromm prev state = learned_domain_next_state_predicates - prev_state_predicates
            # TP = |(original effects) intersection (learned domain effects fromm prev state)|
            # FP = |(learned domain effects fromm prev state) - (original effects)|
            # FN = |(original effects)- (learned domain effects fromm prev state)|
            num_true_positives[executed_action.name] += len(original_effects.intersection(learned_effects))
            num_false_positives[executed_action.name] += len(learned_effects.difference(original_effects))
            num_false_negatives[executed_action.name] += len(original_effects.difference(learned_effects))

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
        self.logger.debug("Starting to calculate the Encoded semantic preconditions performance")
        for index, observation in enumerate(self.dataset_observations):
            observation_objects = observation.grounded_objects
            for component in observation.components:
                action = component.grounded_action_call

                true_positive, false_positive, false_negative = self._calculate_action_applicability_rate(
                    action, learned_domain_path, component.previous_state, observation_objects, learned_domain
                )

                num_true_positives[action.name] += true_positive
                num_false_positives[action.name] += false_positive
                num_false_negatives[action.name] += false_negative

        return _calculate_precision_recall(num_false_negatives, num_false_positives, num_true_positives,
                                           list(self.model_domain.actions.keys()))
