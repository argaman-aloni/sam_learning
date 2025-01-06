from typing import List, Tuple, Dict, Set, Union, Hashable
from pddl_plus_parser.models import Observation, Predicate, ActionCall, State, Domain, ObservedComponent, SignatureType, \
    PDDLType
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
            self.parent[irep] = jrep
            # Increment the result tree's rank by 1
            self.rank[jrep] += 1

class ExtendedSamLearner(SAMLearner):
    """An extension to SAM That can learn in cases of non-injective matching results."""


    possible_effect: dict[str, set[Predicate]]
    cnf_eff: dict[str, And[Or[Var]]]
    cnf_eff_as_set: dict[str, set[Or[Var]]]
    vars_to_forget: dict[str, set[Predicate]]
    def __init__(self,
                 partial_domain: Domain,
                 negative_preconditions_policy: NegativePreconditionPolicy = NegativePreconditionPolicy.soft):
        super().__init__(partial_domain=partial_domain,
                         is_esam=True,
                         negative_preconditions_policy=negative_preconditions_policy)
        self.possible_effect = dict()
        self.cnf_eff_as_set = dict()
        self.vars_to_forget = dict()

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
        lifted_add_effects, lifted_delete_effects = self._handle_action_effects(
            grounded_action, previous_state, next_state)
        lifted_effects = set(lifted_add_effects).union(lifted_delete_effects)
        self.possible_effect[observed_action.name] = set(lifted_effects)
        add_grounded_effects, del_grounded_effects = extract_effects(previous_state, next_state)
        self.cnf_eff_as_set[observed_action.name] = set()

        # add 'Or' clauses to set of 'Or' clauses
        for grounded_effect in add_grounded_effects.union(del_grounded_effects):
            c_eff: list[Var] = list()
            possible_literals = self.matcher.match_predicate_to_action_literals(grounded_effect, grounded_action)
            if len(possible_literals) > 0:
                c_eff.extend([Var(possible_literals) for possible_literals in possible_literals])

            self.cnf_eff_as_set[observed_action.name].add(Or(c_eff))
        grounded_not_effect = extract_not_effects(previous_state, next_state)
        lifted_not_eff = self.matcher.get_possible_literal_matches(grounded_action, list(grounded_not_effect))
        self.vars_to_forget[observed_action.name] = set(lifted_not_eff)

        self.observed_actions.append(observed_action.name)
        self.logger.debug(f"Finished adding the action {grounded_action.name}.")

    def update_action(
            self, grounded_action: ActionCall, previous_state: State, next_state: State) -> None:
        """Create a new action in the domain.

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
        lifted_add_effects, lifted_delete_effects = self._handle_action_effects(
            grounded_action, previous_state, next_state)
        lifted_effects = set(lifted_add_effects).union(lifted_delete_effects)
        self.possible_effect[observed_action.name].update(lifted_effects)
        add_grounded_effects, del_grounded_effects = extract_effects(previous_state, next_state)
        for grounded_effect in add_grounded_effects.union(del_grounded_effects):
            c_eff: list[Var] = list()
            possible_literals = self.matcher.match_predicate_to_action_literals(grounded_effect, grounded_action)
            if len(possible_literals) > 0:
                c_eff.extend([Var(possible_literals) for possible_literals in possible_literals])

            self.cnf_eff_as_set[grounded_action.name].add(Or(c_eff))

        grounded_not_effect = extract_not_effects(previous_state, next_state)
        lifted_not_eff = self.matcher.get_possible_literal_matches(grounded_action, list(grounded_not_effect))
        self.vars_to_forget[observed_action.name].update(lifted_not_eff)

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

    def build_cnf_formulas(self):
        # build initial deducted cnf sentence for each action
        self.cnf_eff = {k: And(v) for k, v in self.cnf_eff_as_set.items()}
        for action_name in self.cnf_eff.keys():
            # forget all effect who are surely not an effect
            self.cnf_eff[action_name] = self.cnf_eff[action_name].forget(self.vars_to_forget[action_name])
            # minimize sentence to prime implicates
            self.cnf_eff[action_name] = self.cnf_eff[action_name].implicates()


    def create_proxy_actions(self, action_name: str):
        proxy_preconds = 0
        proxy_effects = 1

        proxies = list()
        for model in self.cnf_eff[action_name].models():
            effect = set()
            preconds_to_add = set()
            for k, v in model.items():
                predicate = self.hashable_to_predicate(k)
                if v:
                    effect.add(predicate)
                else:
                    preconds_to_add.add(predicate)
            if self.negative_preconditions_policy == NegativePreconditionPolicy.hard and any(
                    not p.is_positive for p in preconds_to_add):
                continue

            proxies.append((preconds_to_add, effect, model))

        if len(proxies) == 1:
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
                effects: set[Predicate] = proxy[proxy_effects]

                #initialize action
                self.partial_domain.actions[name] = LearnerAction(name, signature=signature)
                # set effects
                self.partial_domain.actions[name].discrete_effects = effects
                #update union all proxy cnf_eff negative literals to be preconditions
                self.partial_domain.actions[name].preconditions.root.operands = preconds
                proxy_number+=1
            self.partial_domain.actions.pop(action_name)

    def handle_negative_preconditions_policy(self):
        for action_name in self.partial_domain.actions.keys():
            if self.negative_preconditions_policy != NegativePreconditionPolicy.no_remove:
                preconds_to_keep = set()
                for precond in self.partial_domain.actions[action_name].preconditions.root.operands:
                    if precond.is_positive:
                        preconds_to_keep.add(precond)
                self.partial_domain.actions[action_name].preconditions.root.operands = preconds_to_keep

    def learn_action_model(self, observations: List[Observation]) -> Tuple[LearnerDomain, Dict[str, str]]:
        """Learn the SAFE action model from the input trajectories.

        :param observations: the list of trajectories that are used to learn the safe action model.
        :return: a domain containing the actions that were learned.
        """
        self.logger.info("Starting to learn the action model!")
        for observation in observations:
            self.current_trajectory_objects = observation.grounded_objects
            for component in observation.components:
                self.handle_single_trajectory_component(component)
        # todo add logic of action creation using nnf models for creating proxy actions
        # note that creating will reset all of the actions effects and precondition due to cnf solver usage.
        self.logger.debug("creating updated actions")
        #build all cnf sentences for action's effects
        self.handle_negative_preconditions_policy()
        self.build_cnf_formulas()
        for action_name in self.observed_actions:
            self.create_proxy_actions(action_name)
        learning_report = self._construct_learning_report()
        return self.partial_domain, learning_report

    # @staticmethod
    # def minimize_parameters(model_dict: dict[LearnedLiftedFluent, bool],
    #                         act_num_of_param: int) -> dict[int, int]:
    #     """
    #     the method computes the minimization of parameter list
    #     Args:
    #         act_num_of_param: the length of action's param's list
    #         model_dict: represents the cnf, maps each literal to its grounded value
    #     Returns:
    #         a dictionary mapping each original param act ind_ to the new actions minimized parameter list
    #     """
    #     # make a table that determines if an act ind 'i' is an effect in all occurrences of F, nad is bound to index 'j'
    #     #  in F, minimize i with all indexes t who are bound to 'j' in F in all true occurrences of F
    #
    #     ret_dict: dict[int, int] = dict()
    #     if len(model_dict.keys()) == 0:
    #         return ret_dict
    #
    #     ind_occ: dict[str, list[set[int]]] = dict()
    #     for f in model_dict.keys():
    #         ind_occ[f.name] = (list())
    #         for _ in range(len(f.param_act_inds)):
    #             ind_occ[f.name].append(set())
    #
    #     not_to_minimize: set[int] = set()
    #
    #     for f, val in model_dict.items():
    #         if not val:
    #             not_to_minimize.update(f.param_act_inds)
    #
    #     ind_sets = DisjointSet(act_num_of_param)
    #     for f, val in model_dict.items():
    #         for i in range(len(f.param_act_inds)):
    #             if f.param_act_inds[i] not in not_to_minimize:
    #                 ind_occ[f.name][i].add(f.param_act_inds[i])
    #
    #     for i in set(range(act_num_of_param)).difference(not_to_minimize):
    #         for f, set_list in ind_occ.items():
    #             for sett in set_list:
    #                 if i in sett:
    #                     for j in sett:
    #                         ind_sets.union_by_rank(i, j)
    #
    #     ugly_inds: list[int] = list({ind_sets.find(i) for i in range(act_num_of_param)})
    #     ugly_inds.sort()
    #     for i in range(act_num_of_param):
    #         ret_dict[i] = ugly_inds.index(ind_sets.find(i))
    #
    #     return ret_dict
    #
    # @staticmethod
    # def modify_fluent_params(fluents: set[LearnedLiftedFluent],
    #                          param_dict: dict[int, int]) -> set[LearnedLiftedFluent]:
    #
    #     new_set: set[LearnedLiftedFluent] = set()
    #     for f in fluents:
    #         new_f = LearnedLiftedFluent(name=f.name, param_sorts=f.param_sorts,
    #                                          param_act_inds=[param_dict[i] for i in f.param_act_inds])
    #         new_set.add(new_f)
    #     return new_set

    @staticmethod
    def hashable_to_predicate(hashable: Hashable) -> Predicate:
        """Converts a hashable object back into a Predicate instance, ignoring types."""
        # Convert the hashable object to a string
        pred_str = str(hashable)

        # Handle negation
        is_positive = not pred_str.startswith("(not ")
        if not is_positive:
            pred_str = pred_str[5:-1]  # Remove "(not (" and ")"

        # Strip parentheses and split the string
        pred_str = pred_str.strip("()")
        parts = pred_str.split()

        # First part is the predicate name
        name = parts[0]
        params = list()
        parts = parts[1:]
        for i in range(len(parts)-2):
            params.append((parts[i], parts[i+2]))
            i+=2

        # Remaining parts are parameters
        signature = {param[0]: PDDLType(param[1]) for param in params if param}

        return Predicate(name=name, signature=signature, is_positive=is_positive)
