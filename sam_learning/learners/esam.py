import logging
from typing import List, Tuple, Dict, Hashable, Set

from conda_build.render import actions_to_pins
from nnf import And, Or, Var
from pddl_plus_parser.lisp_parsers.parsing_utils import parse_predicate_from_string
from pddl_plus_parser.models import Observation, Predicate, ActionCall, State, Domain, ObservedComponent, SignatureType, PDDLType, GroundedPredicate
from scipy.cluster.hierarchy import DisjointSet

from sam_learning.core import extract_effects, LearnerDomain, LearnerAction, extract_not_effects
from sam_learning.learners.sam_learning import SAMLearner
from utilities import NegativePreconditionPolicy


class ExtendedSamLearner(SAMLearner):
    """An extension to SAM That can learn in cases of non-injective matching results."""

    cnf_eff: Dict[str, And[Or[Var]]]
    action_effects_cnfs: Dict[str, Set[Or[Var]]]
    cannot_be_effects: Dict[str, Set[str]]

    def __init__(
        self, partial_domain: Domain, negative_preconditions_policy: NegativePreconditionPolicy = NegativePreconditionPolicy.hard_but_allow_proxy
    ):
        super().__init__(partial_domain=partial_domain, negative_preconditions_policy=negative_preconditions_policy)
        self.logger = logging.getLogger(__name__)
        self.cnf_eff = {}
        self.action_effects_cnfs = {action_name: set() for action_name in self.partial_domain.actions.keys()}
        self.cannot_be_effects = {action_name: set() for action_name in self.partial_domain.actions.keys()}

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
            if self.negative_preconditions_policy != NegativePreconditionPolicy.hard:
                # create negated precondition (true-> false, false -> true)
                predicate_negated_copy = predicate.copy(is_negated=True)

                # check for negated precondition is in action preconditions to avoid contradictions
                if predicate_negated_copy in self.partial_domain.actions[action_name].preconditions.root.operands:
                    # negated precondition found, therefore contradiction, return false.
                    return True

            else:  # if no negative preconditions are allowed by policy
                if not predicate.is_positive:
                    # predicate is negative, no neg preconds are allowed, therefore contradiction
                    return True

        return False

    def construct_proxy_action(
        self,
        action_name: str,
        proxy_missing_precondition: Set[Predicate],
        proxy_effects: Set[Predicate],
        modified_parameter_mapping_dict: Dict[str, str],
        proxy_number: int,
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
        proxy_action_name = f"{action_name}_{proxy_number}"
        self.logger.debug(f" constructing proxy action: {proxy_action_name}")
        signature = self.partial_domain.actions[action_name].signature
        preconds: Set[Predicate] = proxy_missing_precondition
        preconds.update(p for p in self.partial_domain.actions[action_name].preconditions.root.operands if isinstance(p, Predicate))

        # maps each param to its set representative param
        # use new mapping to modify effects bindings
        effects: Set[Predicate] = modify_predicate_signature(proxy_effects, modified_parameter_mapping_dict)
        # use new mapping to modify preconditions bindings
        preconds = modify_predicate_signature(preconds, modified_parameter_mapping_dict)

        # use reverse the new mapping for min minimizing action's signature
        reversed_proxy_signature_modified_param_dict: Dict[str, str] = {
            new_repr: old_repr for old_repr, new_repr in modified_parameter_mapping_dict.items()
        }

        new_signature = {parameter: signature[parameter] for parameter in reversed_proxy_signature_modified_param_dict.keys()}
        proxy_action = LearnerAction(proxy_action_name, signature=new_signature)
        proxy_action.discrete_effects = effects
        proxy_action.preconditions.root.operands = preconds
        self.logger.debug(f" finished construction of proxy action: {proxy_action_name}")
        return proxy_action

    def handle_lifted_action_instances(self, action_name: str, action_proxies_data: List[Tuple[Set[Predicate], Set[Predicate], Dict[str, str]]]):
        """
        adds the lifted action additional information to the partial domain, if proxys are needed, the adds proxys to
        the partial domain.
        Args:
            action_name: the name of the lifted action.
            action_proxies_data: list of Tuples where each tuple has preconditions, effect and dictionary.
        """
        proxy_preconds = 0
        proxy_effects = 1
        proxy_model_dict = 2

        if len(action_proxies_data) == 1:  # to avoid index in the action name if not needed
            self.logger.debug(f" updating single instance of action: {action_name}")
            preconds: Set[Predicate] = action_proxies_data[0][proxy_preconds]
            self.partial_domain.actions[action_name].preconditions.root.operands.update(preconds)
            self.partial_domain.actions[action_name].discrete_effects = action_proxies_data[0][proxy_effects]

        elif len(action_proxies_data) > 1:
            self.logger.debug(f" creating proxy actions for action: {action_name}")
            proxy_number = 1
            for proxy_data in action_proxies_data:
                # unpack tuple fields to get properties of proxy action into arguments of action constructor
                new_proxy = self.construct_proxy_action(
                    action_name=action_name,
                    proxy_missing_precondition=proxy_data[proxy_preconds],
                    proxy_effects=proxy_data[proxy_effects],
                    modified_parameter_mapping_dict=proxy_data[proxy_model_dict],
                    proxy_number=proxy_number,
                )

                # add proxy action to Learned domain action model
                self.partial_domain.actions[new_proxy.name] = new_proxy
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
                negative_assigned_predicates = set(pred for pred in model.keys() if not model[pred])
                positive_assigned_predicates = set(model.keys()).difference(negative_assigned_predicates)
                # check for contradiction before running
                self.logger.debug(f"checking for contradiction action: {action_name} cnf assignment.")
                if self.is_proxy_contradiction(negative_assigned_predicates, action_name):
                    self.logger.debug(f"contradiction found in action: {action_name} cnf assignment, skipping assignment.")
                    continue

                self.logger.debug(f"No contradiction found for action: {action_name} cnf assignment.")
                # take all positive assigned predicates and construct a Predicate with hashable instance from provided model
                # add them as effects
                effect: Set[Predicate] = {
                    parse_predicate_from_string(str(parameter_bound_literal), self.partial_domain.types)
                    for parameter_bound_literal in positive_assigned_predicates
                }

                # take all negative assigned predicates and construct a Predicate with hashable instance from provided model
                # add them as preconditions
                preconds_to_add: Set[Predicate] = {
                    parse_predicate_from_string(str(parameter_bound_literal), self.partial_domain.types)
                    for parameter_bound_literal in negative_assigned_predicates
                }

                # assemble proxy info
                self.logger.debug(f"assigning new representatives for parameters list of action {action_name}")
                proxy_signature_modified_param_dict = get_minimize_parameters_equality_dict(
                    model_dict=model, act_signature=self._action_signatures[action_name], domain_types=self.partial_domain.types
                )

                action_proxies.append((preconds_to_add, effect, proxy_signature_modified_param_dict))

            self.handle_lifted_action_instances(action_name, action_proxies)

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

        # note that creating will reset all the actions effects and precondition due to cnf solver usage.

        # build all cnf sentences for action's effects
        super().handle_negative_preconditions_policy()
        self._remove_unobserved_actions_from_partial_domain()
        self.logger.debug(f"building domain actions CNF formulas")
        self.build_cnf_formulas()
        self.construct_safe_actions()
        self.handle_negative_preconditions_policy()
        learning_report = self._construct_learning_report()
        return self.partial_domain, learning_report


def get_minimize_parameters_equality_dict(model_dict: Dict[Hashable, bool], act_signature: SignatureType, domain_types) -> Dict[str, str]:
    """
    the method computes the minimization of parameter list
    Args:
        model_dict (Dict[Hashable, bool]): represents the cnf, maps each literal to its value in the cnf formula solution
        act_signature (SignatureType): the signature of the action
        domain_types (Dict[str, PddlType]): the domain types
    Returns:
        a dictionary mapping each original param act ind_ to the new actions minimized parameter list
    """
    # make a table that determines if an act ind 'i' is an effect in all occurrences of F, nad is bound to index 'j'
    #  in F, minimize i with all indexes t who are bound to 'j' in F in all true occurrences of F

    # reduce the problem to instance of macq, transform params from str too int by index in action
    # transformation is for deciding what parameters to reduce by order in action signature
    new_model_dict: Dict[Predicate, bool] = {parse_predicate_from_string(str(h), domain_types): v for h, v in model_dict.items()}

    param_index_in_action = {param: index for index, param in enumerate(act_signature.keys())}
    reversed_param_index_in_action = {index: param for param, index in param_index_in_action.items()}

    param_index_in_predicate: Dict[Predicate, Dict[str, int]] = {}
    for predicate in new_model_dict.keys():
        param_index_in_predicate[predicate] = dict()
        for index, param in enumerate(predicate.signature.keys()):
            param_index_in_predicate[predicate][param] = index

    predicate_to_param_act_inds: Dict[Predicate, list[int]] = {}
    for predicate in new_model_dict.keys():
        predicate_to_param_act_inds[predicate] = []
        for param in predicate.signature.keys():
            predicate_to_param_act_inds[predicate].append(param_index_in_action[param])

    # start algorithm of parameters equality check
    if len(new_model_dict.keys()) == 0:
        return {}

    ind_occ: Dict[str, List[set[int]]] = {}
    for predicate in new_model_dict.keys():
        ind_occ[predicate.name] = []
        for _ in range(len(predicate_to_param_act_inds[predicate])):
            ind_occ[predicate.name].append(set())

    not_to_minimize: Set[int] = set()

    for predicate, val in new_model_dict.items():
        if not val:
            not_to_minimize.update(predicate_to_param_act_inds[predicate])

    ind_sets = DisjointSet(i for i in range(len(param_index_in_action.keys())))
    for predicate, val in new_model_dict.items():
        for i in range(len(predicate_to_param_act_inds[predicate])):
            if predicate_to_param_act_inds[predicate][i] not in not_to_minimize:
                ind_occ[predicate.name][i].add(predicate_to_param_act_inds[predicate][i])

    for f, set_list in ind_occ.items():
        for sett in set_list:
            if len(sett) > 0:
                set_as_sorted_list = list(sett)
                set_as_sorted_list.sort()
                i = set_as_sorted_list[0]
                for j in set_as_sorted_list:
                    ind_sets.merge(i, j)

    ret_dict_by_indexes: Dict[int, int] = {}
    ugly_inds: List[int] = list({ind_sets.__getitem__(i) for i in range(len(param_index_in_action.keys()))})
    ugly_inds.sort()

    for i in range(len(param_index_in_action.keys())):
        ret_dict_by_indexes[i] = ugly_inds.index(ind_sets.__getitem__(i))

    # transform all indexes back to  str to fit sam learning conventions
    ret_dict_by_param_name: Dict[str, str] = {
        reversed_param_index_in_action[k1]: reversed_param_index_in_action[k2] for k1, k2 in ret_dict_by_indexes.items()
    }

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
