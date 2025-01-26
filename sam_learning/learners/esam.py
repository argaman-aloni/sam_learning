from typing import List, Tuple, Dict, Hashable

from pddl_plus_parser.lisp_parsers.parsing_utils import parse_predicate_from_string
from pddl_plus_parser.models import Observation, Predicate, ActionCall, State, Domain, ObservedComponent, SignatureType, \
    PDDLType, GroundedPredicate
from sam_learning.core import  extract_effects, LearnerDomain, LearnerAction, extract_not_effects
from sam_learning.learners.sam_learning import SAMLearner
from nnf import And, Or, Var

from utilities import NegativePreconditionPolicy


class DisjointSet:  # this class was taken from geeksForGeeks
    def __init__(self, size):
        self.parent = [i for i in range(size)]
        self.rank = [0] * size

    # Function to find the representative (or the root node) of a set
    def find(self, i):
        # If the index 'i' is not the representative of its set, recursively find the representative
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])  # Path compression
        return self.parent[i]

    def union(self, i, j):
        irep = self.find(i)
        jrep = self.find(j)
        if irep == jrep:
            return
        else:
            self.parent[jrep] = irep


    # Unites the set that includes i and the set that includes j by rank
    def union_by_rank(self, i, j):
        # Find the representatives (or the root nodes) for the set that includes i and j
        irep = self.find(i)
        jrep = self.find(j)

        # Elements are in the same set, no need to unite anything
        if irep == jrep:
            return

        # Get the rank of i's tree
        irank = self.rank[irep]

        # Get the rank of j's tree
        jrank = self.rank[jrep]

        # If i's rank is less than j's rank
        if irank < jrank:
            # Move i under j
            self.parent[irep] = jrep
        # Else if j's rank is less than i's rank
        elif jrank < irank:
            # Move j under i
            self.parent[jrep] = irep
        # Else if their ranks are the same
        else:
            # Move i under j (doesn't matter which one goes where)
            self.parent[jrep] = irep
            # Increment the result tree's rank by 1
            self.rank[irep] += 1

class ExtendedSamLearner(SAMLearner):
    """An extension to SAM That can learn in cases of non-injective matching results."""


    possible_effect: dict[str, set[Predicate]]
    cnf_eff: dict[str, And[Or[Var]]]
    cnf_eff_as_set: dict[str, set[Or[Var]]]
    vars_to_forget: dict[str, set[str]]
    def __init__(self,
                 partial_domain: Domain,
                 negative_preconditions_policy: NegativePreconditionPolicy = NegativePreconditionPolicy.hard_but_allow_proxy):
        super().__init__(partial_domain=partial_domain,
                         is_esam=True,
                         negative_preconditions_policy=negative_preconditions_policy)
        self.possible_effect = dict()
        self.cnf_eff_as_set = dict()
        self.vars_to_forget = dict()

    def get_is_eff_clause_for_predicate(self,
                                        grounded_action: ActionCall,
                                        grounded_effect: GroundedPredicate) ->Or[Var]:
        """
        Get the is effect clause for a given predicate by matching it to action literals.

        Parameters:
            grounded_action (ActionCall): The action call related to the grounded effect.
            grounded_effect (GroundedPredicate): The grounded predicate to match.

        Returns:
            Or[Var]: The Or clause composed of is_effect for the given predicate.

        """
        c_eff: list[Var] = list()
        possible_literals = self.matcher.match_predicate_to_action_literals(grounded_effect, grounded_action)
        if len(possible_literals) > 0:
            c_eff.extend([Var(possible_literals) for possible_literals in possible_literals])
        return  Or(c_eff)

    def get_surely_not_eff(self,
                           previous_state: State,
                           next_state: State,
                           grounded_action: ActionCall) -> set[Predicate]:
        """
        Return the set of predicates representing the negative effects caused by the action between the previous state and the next state.

        Parameters:
            previous_state (State): The previous state before the action is taken.
            next_state (State): The state resulting from taking the action.
            grounded_action (ActionCall): The grounded action that was executed.

        Returns:
            set[Predicate]: A set of predicates that cannot be an effect.
        """
        grounded_not_effect = extract_not_effects(previous_state, next_state)
        lifted_not_eff = self.matcher.get_possible_literal_matches(grounded_action, list(grounded_not_effect))
        return set(lifted_not_eff)
    def add_new_action(self, grounded_action: ActionCall, previous_state: State, next_state: State) -> None:
        """Create a new action in the domain.

        :param grounded_action: the grounded action that was executed according to the trajectory.
        :param previous_state: the state that the action was executed on.
        :param next_state: the state that was created after executing the action on the previous
        """
        self.logger.info(f"Adding the action {str(grounded_action)} to the domain.")
        # adding the preconditions each predicate is grounded in this stage.
        observed_action = self.partial_domain.actions[grounded_action.name]
        super()._add_new_action_preconditions(grounded_action)

        # handling effects
        add_grounded_effects, del_grounded_effects = extract_effects(previous_state, next_state)
        self.cnf_eff_as_set[observed_action.name] = set()

        # add 'Or' clauses to set of 'Or' clauses
        for grounded_effect in add_grounded_effects.union(del_grounded_effects):
            or_clause = self.get_is_eff_clause_for_predicate(grounded_action, grounded_effect)
            self.cnf_eff_as_set[grounded_action.name].add(or_clause)

        # extract predicated who are surely not an effect
        not_eff_set = self.get_surely_not_eff(next_state, grounded_action)
        self.vars_to_forget[observed_action.name] = {eff.untyped_representation for eff in not_eff_set}
        self.observed_actions.append(observed_action.name)
        self.logger.debug(f"Finished adding the action {grounded_action.name}.")

    def update_action(
            self, grounded_action: ActionCall, previous_state: State, next_state: State) -> None:
        """updates an existing action in the domain based on a transition.

        :param grounded_action: the grounded action that was executed according to the trajectory.
        :param previous_state: the state that the action was executed on.
        :param next_state: the state that was created after executing the action on the previous
            state.
        """
        action_name = grounded_action.name
        observed_action = self.partial_domain.actions[action_name]
        # handle preconditions
        super()._update_action_preconditions(grounded_action)
        # handle effects
        add_grounded_effects, del_grounded_effects = extract_effects(previous_state, next_state)
        # add 'Or' clauses to set of 'Or' clauses
        for grounded_effect in add_grounded_effects.union(del_grounded_effects):
            or_clause = self.get_is_eff_clause_for_predicate(grounded_action, grounded_effect)
            self.cnf_eff_as_set[grounded_action.name].add(or_clause)

        not_eff_set = self.get_surely_not_eff(previous_state, next_state, grounded_action)
        self.vars_to_forget[observed_action.name] = not_eff_set

        # add all predicates who are surely not an effect for future
        self.logger.debug(f"Done updating the action cnf formulas - {grounded_action.name}")

    def handle_single_trajectory_component(self, component: ObservedComponent) -> None:
        """Handles a single trajectory component as a part of the learning process.

        :param component: the trajectory component that is being handled at the moment.
        """
        previous_state = component.previous_state
        grounded_action = component.grounded_action_call
        next_state = component.next_state
        action_name = grounded_action.name

        self.triplet_snapshot.create_triplet_snapshot(
            previous_state=previous_state, next_state=next_state, current_action=grounded_action,
            observation_objects=self.current_trajectory_objects)
        if action_name in self.observed_actions:
            self.update_action(grounded_action, previous_state, next_state)

        else:
            self.add_new_action(grounded_action, previous_state, next_state)

    def build_cnf_formulas(self) -> None:
        """
        for each action, builds the effect cnf formula
        """
        # build initial deducted cnf sentence for each action
        self.cnf_eff = {k: And(v) for k, v in self.cnf_eff_as_set.items()}
        for action_name in self.cnf_eff.keys():
            # forget all effect who are surely not an effect
            self.cnf_eff[action_name] = self.cnf_eff[action_name].forget(self.vars_to_forget[action_name])
            # minimize sentence to prime implicates
            self.cnf_eff[action_name] = self.cnf_eff[action_name].implicates()


    def create_lifted_action_data(self, action_name: str) -> None:
        """
        creates and adds the proxy action by its name in the domain to the learned domain.
        Args:
            action_name: the name of the action to build proxys.
        """

        act_signature = self.partial_domain.actions[action_name].signature

        proxies = list()
        for model in self.cnf_eff[action_name].models():
            effect = set()
            preconds_to_add = set()
            is_skip_proxy = False
            for k, v in model.items():
                predicate = parse_predicate_from_string(str(k), self.partial_domain.types)
                if v:
                    effect.add(predicate)

                # handle precondition additions
                else:  # check for contradiction
                    if self.negative_preconditions_policy != NegativePreconditionPolicy.hard:
                        predicate_opposite_copy = predicate.copy()
                        #create negated precondition (true-> false, false -> true)
                        predicate_opposite_copy.is_positive = not predicate_opposite_copy.is_positive

                        # check for negated precond in preconds  to avoid contradictions
                        if self.partial_domain.actions[action_name].preconditions.root.operands.__contains__(
                            predicate_opposite_copy):
                            is_skip_proxy = True
                            break

                    else:  # if no negative preconditions are allowed by policy, so skip proxy
                        if not predicate.is_positive:
                            is_skip_proxy = True
                            break

                    # all ok add it to preconds
                    preconds_to_add.add(predicate)

            # check to see if the action needs to be skipped
            if is_skip_proxy:
                continue

            # assemble proxy info
            proxy_signature_modified_param_dict = get_minimize_parameters_equality_dict(model_dict=model,
                                                                                act_signature=act_signature,
                                                                                domain_types=self.partial_domain.types)

            proxies.append((preconds_to_add, effect, proxy_signature_modified_param_dict))

        self.add_lifted_action_instance(action_name, proxies)


    def add_lifted_action_instance(self,
                                   action_name: str,
                                   proxies: list[tuple[set[Predicate], set[Predicate], dict[str, str]]]):
        """
        adds the lifted action additional information to the partial domain, if proxys are needed, the adds proxys to
        the partial domain.
        Args:
            action_name: the name of the lifted action.
            proxies: the proxies of the lifted action data, if len 1, no proxy needed.
        """
        proxy_preconds = 0
        proxy_effects = 1
        proxy_model_dict = 2

        if len(proxies) == 1: # to avoid index in the action name if not needed
            self.partial_domain.actions[action_name].discrete_effects = proxies[0][proxy_effects]
            preconds: set[Predicate] = proxies[0][proxy_preconds]
            self.partial_domain.actions[action_name].preconditions.root.operands.update(preconds)

        elif len(proxies) > 1:
            proxy_number = 1
            for proxy in proxies:
                # unpack tuple fields to get properties of proxy action
                name = f"{action_name}_{proxy_number}"
                signature = self.partial_domain.actions[action_name].signature
                preconds: set[Predicate] = proxy[proxy_preconds]
                preconds.update(p for p in self.partial_domain.actions[action_name].preconditions.root.operands
                                if isinstance(p,Predicate))

                proxy_signature_modified_param_dict = proxy[proxy_model_dict]
                effects: set[Predicate] = modify_predicate_signature(proxy[proxy_effects],
                                                                     proxy_signature_modified_param_dict)
                preconds = modify_predicate_signature(preconds, proxy_signature_modified_param_dict)
                reversed_proxy_signature_modified_param_dict: dict[str, str] = {
                    v: k for k, v in proxy_signature_modified_param_dict.items()}

                new_signature = {k: signature[k] for k in reversed_proxy_signature_modified_param_dict.keys()}

                #initialize action
                self.partial_domain.actions[name] = LearnerAction(name, signature=new_signature)
                # set effects
                self.partial_domain.actions[name].discrete_effects = effects
                #update union all proxy cnf_eff negative literals to be preconditions
                self.partial_domain.actions[name].preconditions.root.operands = preconds
                proxy_number+=1
            self.partial_domain.actions.pop(action_name)

    def esam_handle_negative_preconditions_policy(self):
        for action_name in self.partial_domain.actions.keys():
            if self.negative_preconditions_policy in [NegativePreconditionPolicy.hard,
                                                      NegativePreconditionPolicy.hard_but_allow_proxy]:
                preconds_to_keep = set()
                for precond in self.partial_domain.actions[action_name].preconditions.root.operands:
                    if precond.is_positive:
                        preconds_to_keep.add(precond)
                self.partial_domain.actions[action_name].preconditions.root.operands = preconds_to_keep

    def handle_observations(self, observations: list[Observation]):
        """
        Handles observations from input, invokes learning methods prior to deducting effects by cnf.

        Parameters:
        observations (list[Observation]): List of Observation to learn from.
        """
        for observation in observations:
            self.current_trajectory_objects = observation.grounded_objects
            for component in observation.components:
                self.handle_single_trajectory_component(component)
    def learn_action_model(self, observations: List[Observation]) -> Tuple[LearnerDomain, Dict[str, str]]:
        """Learn the SAFE action model from the input trajectories.

        :param observations: the list of trajectories that are used to learn the safe action model.
        :return: a domain containing the actions that were learned.
        """
        self.logger.info("Starting to learn the action model!")

        self._complete_possibly_missing_actions()
        self.handle_observations(observations)
        # note that creating will reset all the actions effects and precondition due to cnf solver usage.
        self.logger.debug("creating updated actions")
        #build all cnf sentences for action's effects
        self.esam_handle_negative_preconditions_policy()
        self._remove_unobserved_actions_from_partial_domain()
        self.build_cnf_formulas()
        for action_name in self.observed_actions:
            self.create_lifted_action_data(action_name)

        self.handle_negative_preconditions_policy()
        learning_report = self._construct_learning_report()
        return self.partial_domain, learning_report


def get_minimize_parameters_equality_dict(model_dict: dict[Hashable, bool],
                                          act_signature: SignatureType,
                                          domain_types) -> dict[str, str]:
    """
    the method computes the minimization of parameter list
    Args:
        act_signature: the signature of the action
        model_dict: represents the cnf, maps each literal to its value in the cnf formula solution
        domain_types: the domain types
    Returns:
        a dictionary mapping each original param act ind_ to the new actions minimized parameter list
    """
    # make a table that determines if an act ind 'i' is an effect in all occurrences of F, nad is bound to index 'j'
    #  in F, minimize i with all indexes t who are bound to 'j' in F in all true occurrences of F

    # reduce the problem to instance of macq, transform params from str too int by index in action
    # transformation is for deciding what parameters to reduce by order in action signature
    new_model_dict: dict[Predicate, bool] = {parse_predicate_from_string(str(h), domain_types): v for h, v in model_dict.items()}
    param_index_in_action = {param: index for index, param in enumerate(act_signature.keys())}
    num_of_act_params = len(param_index_in_action.keys())
    reversed_param_index_in_action = {v: k for k, v in param_index_in_action.items()}

    param_index_in_predicate: dict[Predicate, dict[str, int]] = dict()
    for predicate in new_model_dict.keys():
        param_index_in_predicate[predicate] = dict()
        for index , param in enumerate(predicate.signature.keys()):
            param_index_in_predicate[predicate][param] = index

    predicate_to_param_act_inds: dict[Predicate, list[int]] = dict()
    for predicate in new_model_dict.keys():
        predicate_to_param_act_inds[predicate] = list()
        for param in predicate.signature.keys():
            predicate_to_param_act_inds[predicate].append(param_index_in_action[param])

# start algorithm of parameters equality check
    if len(new_model_dict.keys()) == 0:
        return dict()

    ind_occ: dict[str, list[set[int]]] = dict()
    for predicate in new_model_dict.keys():
        ind_occ[predicate.name] = (list())
        for _ in range(len(predicate_to_param_act_inds[predicate])):
            ind_occ[predicate.name].append(set())

    not_to_minimize: set[int] = set()

    for predicate, val in new_model_dict.items():
        if not val:
            not_to_minimize.update(predicate_to_param_act_inds[predicate])

    ind_sets = DisjointSet(len(param_index_in_action.keys()))
    for predicate, val in new_model_dict.items():
        for i in range(len(predicate_to_param_act_inds[predicate])):
            if predicate_to_param_act_inds[predicate][i] not in not_to_minimize:
                ind_occ[predicate.name][i].add(predicate_to_param_act_inds[predicate][i])

    for i in set(range(num_of_act_params)).difference(not_to_minimize):
        for f, set_list in ind_occ.items():
            for sett in set_list:
                set_as_sorted_list = list(sett)
                set_as_sorted_list.sort()
                if i in set_as_sorted_list:
                    for j in sett:
                        ind_sets.union_by_rank(i, j)

    ret_dict_by_indexes: dict[int, int] = dict()
    ugly_inds: list[int] = list({ind_sets.find(i) for i in range(len(param_index_in_action.keys()))})
    ugly_inds.sort()

    for i in range(len(param_index_in_action.keys())):
        ret_dict_by_indexes[i] = ugly_inds.index(ind_sets.find(i))

# transform all indexes back to  str to fit sam learning conventions
    ret_dict_by_param_name: [str, str] = {
        reversed_param_index_in_action[k1]: reversed_param_index_in_action[k2]
        for k1, k2 in ret_dict_by_indexes.items()}


    return ret_dict_by_param_name

def modify_predicate_signature(predicates: set[Predicate],
                         param_dict: dict[str, str]) -> set[Predicate]:
    """
    modifies a set of predicates to fit the proxy minimized parameter list if minimization is needed
    """
    new_set: set[Predicate] = set()
    for predicate in predicates:
        new_signature: dict[str, PDDLType] = {
            param_dict[param]: predicate.signature[param] for param in predicate.signature.keys()}
        new_predicate = Predicate(name=predicate.name, signature= new_signature, is_positive=predicate.is_positive)
        new_set.add(new_predicate)
    return new_set