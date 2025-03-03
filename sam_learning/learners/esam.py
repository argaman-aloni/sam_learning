import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict, Hashable, Set, Callable, Optional

from nnf import And, Or, Var
from pddl_plus_parser.lisp_parsers.parsing_utils import parse_predicate_from_string
from pddl_plus_parser.models import Observation, Predicate, ActionCall, State, Domain, ObservedComponent, SignatureType, PDDLType, GroundedPredicate
from scipy.cluster.hierarchy import DisjointSet

from sam_learning.core import extract_effects, LearnerDomain, LearnerAction, extract_not_effects
from sam_learning.learners.sam_learning import SAMLearner
from utilities import NegativePreconditionPolicy


@dataclass
class ProxyActionData:
    preconditions: Set[Predicate]
    effects: Set[Predicate]
    signature: Dict[str, str]


class ExtendedSamLearner(SAMLearner):
    """An extension to SAM That can learn in cases of non-injective matching results."""

    cnf_eff: Dict[str, And[Or[Var]]]
    action_effects_cnfs: Dict[str, Set[Or[Var]]]
    cannot_be_effects: Dict[str, Set[str]]
    encoders: Dict[str, list[Callable[[ActionCall], ActionCall]]]
    decoders: Dict[str, Callable[[ActionCall], ActionCall]]

    def __init__(self, partial_domain: Domain, negative_preconditions_policy: NegativePreconditionPolicy = NegativePreconditionPolicy.hard):
        super().__init__(partial_domain=partial_domain, negative_preconditions_policy=negative_preconditions_policy)
        self.logger = logging.getLogger(__name__)
        self.cnf_eff = {}
        self.action_effects_cnfs = {action_name: set() for action_name in self.partial_domain.actions.keys()}
        self.cannot_be_effects = {action_name: set() for action_name in self.partial_domain.actions.keys()}
        self.encoders = {}
        self.decoders = {}

    def encode(self, action_call: ActionCall) -> List[ActionCall]:
        encoders: List[Callable] = self.encoders[action_call.name]
        possible_action_calls: List[ActionCall] = [encoder(action_call) for encoder in encoders]
        return possible_action_calls

    def decode(self, action_call: ActionCall) -> ActionCall:
        return self.decoders[action_call.name](action_call)

    def _get_is_eff_clause_for_predicate(self, grounded_action: ActionCall, grounded_effect: GroundedPredicate) -> Or[Var]:
        """
        Get the is effect clause for a given predicate by matching it to action literals.

        Parameters:
            grounded_action (ActionCall): The action call related to the grounded effect.
            grounded_effect (GroundedPredicate): The grounded predicate to match.

        Returns:
            Or[Var]: The Or clause composed of is_effect for the given predicate.

        """
        clause_effects: List[Var] = []
        possible_literals = self.matcher.match_predicate_to_action_literals(grounded_effect, grounded_action)

        if len(possible_literals) > 0:
            literal = [Var(possible_literal.untyped_representation) for possible_literal in possible_literals]
            clause_effects.extend(literal)

        return Or(clause_effects)

    def _get_surely_not_eff(self, grounded_action: ActionCall) -> Set[Predicate]:
        """
        Return the set of predicates representing the negative effects caused by the action between the previous state and the next state.

        Parameters:
            grounded_action (ActionCall): The grounded action that was executed.

        Returns:
            set[Predicate]: A set of predicates that cannot be an effect.
        """
        self.logger.info(f"Getting the predicates that are surely not effects for action {grounded_action.name}.")
        not_add_effects, not_delete_effects = extract_not_effects(self.triplet_snapshot.next_state_predicates)
        self.logger.debug("Updating the predicates that cannot be add effects and those that cannot be delete effects.")
        not_lifted_add_effects = self.matcher.get_possible_literal_matches(grounded_action, list(not_add_effects))
        not_lifted_delete_effects = self.matcher.get_possible_literal_matches(grounded_action, list(not_delete_effects))
        return set(not_lifted_add_effects).union(not_lifted_delete_effects)

    def handle_effects(self, previous_state: State, next_state: State, grounded_action: ActionCall):
        # handle effects
        self.logger.debug(f"handling action {grounded_action.name} effects.")
        add_grounded_effects, del_grounded_effects = extract_effects(previous_state, next_state)

        # add 'Or' clauses to set of 'Or' clauses
        self.logger.debug("Adding the effects to the action effects CNF as a new OR clause.")
        for grounded_effect in add_grounded_effects.union(del_grounded_effects):
            or_clause = self._get_is_eff_clause_for_predicate(grounded_action, grounded_effect)
            self.action_effects_cnfs[grounded_action.name].add(or_clause)

        cannot_be_effects = self._get_surely_not_eff(grounded_action)

        self.cannot_be_effects[grounded_action.name].update({eff.untyped_representation for eff in cannot_be_effects})
        self.logger.debug(f"finished handling action {grounded_action.name} effects.")

    def add_new_action(self, grounded_action: ActionCall, previous_state: State, next_state: State) -> None:
        """Create a new action in the domain.

        :param grounded_action: the grounded action that was executed according to the trajectory.
        :param previous_state: the state that the action was executed on.
        :param next_state: the state that was created after executing the action on the previous
        """
        self.logger.info(f"Adding the action {str(grounded_action)} to the domain.")
        # adding the preconditions each predicate is grounded in this stage.
        observed_action = self.partial_domain.actions[grounded_action.name]
        self.observed_actions.append(observed_action.name)
        super()._add_new_action_preconditions(grounded_action)
        self.handle_effects(previous_state, next_state, grounded_action)

    def update_action(self, grounded_action: ActionCall, previous_state: State, next_state: State) -> None:
        """updates an existing action in the domain based on a transition.

        :param grounded_action: the grounded action that was executed according to the trajectory.
        :param previous_state: the state that the action was executed on.
        :param next_state: the state that was created after executing the action on the previous
            state.
        """
        self.logger.debug(f"updating action {str(grounded_action)}.")
        super()._update_action_preconditions(grounded_action)
        self.handle_effects(previous_state, next_state, grounded_action)
        self.logger.debug(f"finished updating action {str(grounded_action)}.")

    def handle_single_trajectory_component(self, component: ObservedComponent) -> None:
        """Handles a single trajectory component as a part of the learning process.

        :param component: the trajectory component that is being handled at the moment.
        """
        previous_state = component.previous_state
        grounded_action = component.grounded_action_call
        next_state = component.next_state
        action_name = grounded_action.name

        self.triplet_snapshot.create_triplet_snapshot(
            previous_state=previous_state, next_state=next_state, current_action=grounded_action, observation_objects=self.current_trajectory_objects
        )
        if action_name in self.observed_actions:
            self.update_action(grounded_action, previous_state, next_state)

        else:
            self.add_new_action(grounded_action, previous_state, next_state)

    def build_cnf_formulas(self) -> None:
        """For each action, builds the effect cnf formula"""
        self.logger.info("Building the CNF formulas for the action effects.")
        # build initial deducted cnf sentence for each action
        self.cnf_eff = {action_name: And(clauses) for action_name, clauses in self.action_effects_cnfs.items()}
        for action_name in self.cnf_eff:
            self.logger.debug(f"building action {action_name} CNF formula.")
            # forget all effect who are surely not an effect
            self.logger.debug("Forgetting the predicates that cannot be effects.")
            self.cnf_eff[action_name] = self.cnf_eff[action_name].forget(self.cannot_be_effects[action_name])
            # minimize sentence to prime implicates
            self.logger.debug("Minimizing the CNF formula to prime implicates, i.e., returning the minimal formula.")
            self.cnf_eff[action_name] = self.cnf_eff[action_name].implicates()

    def is_proxy_contradiction(self, negative_assigned_predicates: Set[Hashable], action_name: str) -> bool:
        """
        Creates and adds the proxy action by its name in the domain to the learned partial-domain.
        Args:
            negative_assigned_predicates: all predicates with negative assignment in cnf satisfying assignment
            action_name: the original action's name.

        Returns:
            True if contradiction is found, False otherwise.
        """
        for parameter_bound_literal in negative_assigned_predicates:
            predicate = parse_predicate_from_string(str(parameter_bound_literal), self.partial_domain.types)
            predicate_negated_copy = predicate.copy(is_negated=True)
            # check for negated precondition is in action preconditions to avoid contradictions
            if predicate_negated_copy in self.partial_domain.actions[action_name].preconditions.root.operands:
                # negated precondition found, therefore contradiction, return false.
                return True

        return False

    def construct_proxy_action(
        self,
        action_name: str,
        proxy_missing_precondition: Set[Predicate],
        proxy_effects: Set[Predicate],
        modified_parameter_mapping_dict: Dict[str, str],
        proxy_number: Optional[int] = None,
    ) -> LearnerAction:
        """
            constructs a proxy action based on information learned.

            Args:
                action_name (str): the name of the Original lifted action.
                proxy_missing_precondition (Set[Predicate]): A unique set of predicates to add for the action's preconditions.
                proxy_effects (Set[Predicate]): set of predicates that are deducted as effects after solving
                 cnf_effect formula.
                modified_parameter_mapping_dict (Dict[str, str]): a dictionary that maps each argument of the original
                  action to its new representative in the signature.
                proxy_number (int): the index of proxy to be generated

            Returns:
                constructed proxy action
        """
        proxy_action_name = f"{action_name}{'_' + str(proxy_number) if proxy_number else ''}"
        self.logger.debug(f"Starting the constructing proxy action: {proxy_action_name}")
        signature = self.partial_domain.actions[action_name].signature
        discrete_effects = modify_predicate_signature(proxy_effects, modified_parameter_mapping_dict)
        # use new mapping to modify preconditions bindings
        preconds = modify_predicate_signature(
            {*proxy_missing_precondition, *self.partial_domain.actions[action_name].preconditions.root.operands}, modified_parameter_mapping_dict
        )

        # use reverse the new mapping for min minimizing action's signature
        reversed_proxy_signature_modified_param_dict: Dict[str, str] = {
            new_repr: old_repr for old_repr, new_repr in modified_parameter_mapping_dict.items()
        }

        new_signature = {parameter: signature[parameter] for parameter in reversed_proxy_signature_modified_param_dict.keys()}
        proxy_action = LearnerAction(proxy_action_name, signature=new_signature)
        proxy_action.discrete_effects = discrete_effects
        proxy_action.preconditions.root.operands = preconds
        self.logger.debug(f"Finished construction of proxy action: {proxy_action_name}")
        return proxy_action

    def handle_lifted_action_instances(self, action_name: str, action_proxies_data: List[ProxyActionData]):
        """
        adds the lifted action additional information to the partial domain, if proxys are needed, the adds proxys to
        the partial domain.
        Args:
            action_name: the name of the lifted action.
            action_proxies_data: list of Tuples where each tuple has preconditions, effect and dictionary.
        """
        self.logger.debug(f"Creating proxy actions for action: {action_name}")
        proxy_number = 1
        for proxy_data in action_proxies_data:
            # unpack tuple fields to get properties of proxy action into arguments of action constructor
            new_proxy = self.construct_proxy_action(
                action_name=action_name,
                proxy_missing_precondition=proxy_data.preconditions,
                proxy_effects=proxy_data.effects,
                modified_parameter_mapping_dict=proxy_data.signature,
                proxy_number=proxy_number,
            )

            # add proxy action to Learned domain action model
            self.partial_domain.actions[new_proxy.name] = new_proxy

            def decoder(proxy_action_call: ActionCall) -> ActionCall:
                original_signature = self.partial_domain.actions[action_name].signature
                proxy_param_list = list(new_proxy.signature.keys())
                proxy_parameter_reversed_map = {new_param: old_param for old_param, new_param in proxy_data.signature.items()}

                grounded_proxy_obj_list: List[str] = proxy_action_call.parameters
                original_action_map = {
                    old_param: grounded_proxy_obj_list[proxy_param_list.index(new_param)]
                    for new_param, old_param in proxy_parameter_reversed_map.keys()
                }

                new_obj_list = [original_action_map[param] for param in original_signature.keys()]

                return ActionCall(action_name, new_obj_list)

            # TODO: complete encoder, figure out how not injective therefore return multiple optional action calls!
            def encoder(original_action_call: ActionCall) -> ActionCall:
                original_signature = self.partial_domain.actions[action_name].signature
                proxy_param_mapping = proxy_data.signature

                return ActionCall("", [])

            self.decoders[new_proxy.name] = decoder
            self.encoders[action_name].append(encoder)
            proxy_number += 1

        # pop original unsafe action from learned Domain action model
        self.partial_domain.actions.pop(action_name)

    def construct_safe_actions(self) -> None:
        """
            Creates and adds all safe instances of action.
            if injective binding assumption holds for at least 1 observation, only 1 instance is initialized in the learned domain
            if injective binding assumption does not hold for all observations of the action,
                proxy actions are created and added to the domain.
        """
        for action_name in self.observed_actions:
            action_proxies = []
            self.logger.debug(f"Going over all the possible assignments that are true after for the action {action_name}.")
            for model in self.cnf_eff[action_name].models():
                # represent the literals that were not selected as effects for the action -- will be added as preconditions
                negative_assigned_predicates = set(pred for pred in model.keys() if not model[pred])
                # represent the literals that were selected as effects for the action
                positive_assigned_predicates = set(model.keys()).difference(negative_assigned_predicates)
                # check for contradiction before running
                self.logger.debug(f"checking for contradiction action: {action_name} cnf assignment.")
                if self.is_proxy_contradiction(negative_assigned_predicates, action_name):
                    self.logger.debug(f"contradiction found in action: {action_name} cnf assignment, skipping assignment.")
                    continue

                self.logger.debug(f"No contradiction found for action: {action_name} cnf assignment.")
                effects: Set[Predicate] = {
                    parse_predicate_from_string(str(parameter_bound_literal), self.partial_domain.types)
                    for parameter_bound_literal in positive_assigned_predicates
                }

                additional_preconditions: Set[Predicate] = {
                    parse_predicate_from_string(str(parameter_bound_literal), self.partial_domain.types)
                    for parameter_bound_literal in negative_assigned_predicates
                }

                # assemble proxy info
                self.logger.debug(f"assigning new representatives for parameters list of action {action_name}")
                proxy_signature_modified_param_dict = minimize_parameters_equality_dict(
                    model_dict=model, act_signature=self._action_signatures[action_name], domain_types=self.partial_domain.types
                )

                action_proxies.append(ProxyActionData(additional_preconditions, effects, proxy_signature_modified_param_dict))

            if len(action_proxies) == 1:
                self.logger.debug(f"No proxy actions needed for action: {action_name}")
                action_data = action_proxies[0]
                self.partial_domain.actions[action_name].preconditions.root.operands.update(action_data.preconditions)
                self.partial_domain.actions[action_name].discrete_effects = action_data.effects

            elif len(action_proxies) > 1:
                self.logger.debug(f"Creating proxy actions for action: {action_name}")
                self.handle_lifted_action_instances(action_name, action_proxies)

            else:
                raise ValueError(f"No valid assignments found for action: {action_name}")

    def learn_action_model(self, observations: List[Observation]) -> Tuple[LearnerDomain, Dict[str, str]]:
        """Learn the SAFE action model from the input trajectories.

        :param observations: the list of trajectories that are used to learn the safe action model.
        :return: a domain containing the actions that were learned.
        """
        self.logger.info("Starting to learn the action model using the ESAM algorithm!")
        self._complete_possibly_missing_actions()
        for observation in observations:
            self.current_trajectory_objects = observation.grounded_objects
            for component in observation.components:
                self.handle_single_trajectory_component(component)

        super().handle_negative_preconditions_policy()
        self._remove_unobserved_actions_from_partial_domain()
        self.logger.debug(f"building domain actions CNF formulas")
        self.build_cnf_formulas()
        self.construct_safe_actions()
        learning_report = self._construct_learning_report()
        return self.partial_domain, learning_report


def minimize_parameters_equality_dict(
    model_dict: Dict[Hashable, bool], act_signature: SignatureType, domain_types: Dict[str, PDDLType]
) -> Dict[str, str]:
    """
    The method computes the minimization of parameter list
    Args:
        model_dict (Dict[Hashable, bool]): represents the cnf, maps each literal to its value in the cnf formula solution
        act_signature (SignatureType): the signature of the action
        domain_types (Dict[str, PddlType]): the domain types
    Returns:
        a dictionary mapping each original param act ind_ to the new actions minimized parameter list
    """
    # make a table that determines if an act ind 'i' is an effect in all occurrences of F, nad is bound to index 'other_param'
    new_model_dict: Dict[Predicate, bool] = {parse_predicate_from_string(str(h), domain_types): v for h, v in model_dict.items()}

    # start algorithm of parameters equality check
    if len(new_model_dict.keys()) == 0:
        return {}

    param_occ: Dict[str, List[set[str]]] = {}
    for predicate in new_model_dict.keys():
        param_occ[predicate.name] = [set() for _ in range(len(predicate.signature.keys()))]

    params_not_to_minimize = set()
    for predicate, is_selected in new_model_dict.items():
        if not is_selected:
            params_not_to_minimize.update(predicate.signature.keys())

    param_eq_sets = DisjointSet(param for param in act_signature.keys())

    for predicate, is_selected in new_model_dict.items():
        for signature_index, parameter_name in enumerate(predicate.signature.keys()):
            if parameter_name not in params_not_to_minimize:
                param_occ[predicate.name][signature_index].add(parameter_name)

    for equality_set_list in param_occ.values():
        for equality_set in equality_set_list:
            if len(equality_set) < 0:
                continue

            sorted_list = [param for param in act_signature.keys() if param in equality_set]
            for param_to_merge in sorted_list:
                param_eq_sets.merge(sorted_list[0], param_to_merge)

    ret_dict_by_param_name = {param: param_eq_sets[param] for param in act_signature.keys()}

    return ret_dict_by_param_name


def modify_predicate_signature(predicates: Set[Predicate], param_dict: Dict[str, str]) -> Set[Predicate]:
    """
    modifies a set of predicates to fit the proxy minimized parameter list if minimization is needed
    """
    new_set: Set[Predicate] = set()
    for predicate in predicates:
        new_signature: Dict[str, PDDLType] = {param_dict[param]: predicate.signature[param] for param in predicate.signature.keys()}
        new_predicate = Predicate(name=predicate.name, signature=new_signature, is_positive=predicate.is_positive)
        new_set.add(new_predicate)

    return new_set
