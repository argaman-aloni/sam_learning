"""Module to learn action models from multi-agent trajectories with joint actions."""
import logging
from typing import Dict, List, NoReturn, Tuple, Set, Optional

from pddl_plus_parser.models import Predicate, Domain, MultiAgentComponent, PDDLObject, NOP_ACTION, \
    MultiAgentObservation, ActionCall, State, GroundedPredicate, JointActionCall

from sam_learning.core import LearnerDomain, extract_effects, LiteralCNF
from sam_learning.learners import SAMLearner


class MultiAgentSAM(SAMLearner):
    """Class designated to learning action models from multi-agent trajectories with joint actions."""
    logger: logging.Logger
    positive_literals_cnf: Dict[str, LiteralCNF]
    negative_literals_cnf: Dict[str, LiteralCNF]
    preconditions_fluent_map: Dict[str, List[str]]
    concurrency_constraint: int

    def __init__(self, partial_domain: Domain, preconditions_fluent_map: Optional[Dict[str, List[str]]] = None,
                 concurrency_constraint: int = 2):
        super().__init__(partial_domain)
        self.logger = logging.getLogger(__name__)
        self.observed_actions = []
        self.positive_literals_cnf = {}
        self.negative_literals_cnf = {}
        self.preconditions_fluent_map = preconditions_fluent_map if preconditions_fluent_map else {}
        self.concurrency_constraint = concurrency_constraint

    def _initialize_cnfs(self) -> NoReturn:
        """Initialize the CNFs for the action model."""
        for predicate in self.partial_domain.predicates.values():
            self.positive_literals_cnf[predicate.untyped_representation] = \
                LiteralCNF(action_names=list(self.partial_domain.actions.keys()))
            self.negative_literals_cnf[predicate.untyped_representation] = \
                LiteralCNF(action_names=list(self.partial_domain.actions.keys()))

    @staticmethod
    def _locate_executing_action(joint_action: JointActionCall) -> List[ActionCall]:
        """Locates the actions that are being executed in the joint action.

        :param joint_action: the joint action to search in.
        :return: the actions that are being executed.
        """
        executing_actions = []
        for action in joint_action.actions:
            if action.name != NOP_ACTION:
                executing_actions.append(action)

        return executing_actions

    def _lift_predicate(self, grounded_predicate: GroundedPredicate, action_call: ActionCall) -> Optional[Predicate]:
        """Lifts a grounded predicate to a lifted predicate.

        :param grounded_predicate: the grounded predicate to lift.
        :param action_call: the action call that the grounded predicate belongs to.
        :return: the lifted predicate if the predicate was lifted, None otherwise.
        """
        lifted_predicate = self.matcher.match_predicate_to_action_literals(grounded_predicate, action_call)
        if lifted_predicate is None:
            return None

        if len(lifted_predicate) > 1:
            self.logger.warning("Not supporting non injective matches.")
            return None

        return lifted_predicate[0]

    def add_not_effect_to_cnf(
            self, executed_action: ActionCall, next_state_predicates: List[GroundedPredicate],
            literals_cnf: Dict[str, LiteralCNF]) -> NoReturn:
        """

        :param executed_action:
        :param next_state_predicates:
        :param literals_cnf:
        :return:
        """
        for predicate in next_state_predicates:
            bounded_lifted_literal = self.matcher.match_predicate_to_action_literals(predicate, executed_action)
            if bounded_lifted_literal is None:
                continue

            if len(bounded_lifted_literal) > 1:
                self.logger.warning("Not supporting non injective matches.")
                continue

            literals_cnf[predicate.lifted_untyped_representation].add_not_effect(
                executed_action.name, bounded_lifted_literal[0])

    def create_fully_observable_predicates(
            self, state: State,
            observed_objects: Dict[str, PDDLObject]) -> Tuple[List[GroundedPredicate], List[GroundedPredicate]]:
        """

        :param state:
        :param observed_objects:
        :return:
        """

    def handle_concurrent_execution(
            self, grounded_effect: GroundedPredicate,
            executing_actions: List[ActionCall], literals_cnf: Dict[str, LiteralCNF]) -> NoReturn:
        """

        :param grounded_effect:
        :param executing_actions:
        :param literals_cnf:
        :return:
        """
        interacting_actions = self._get_interacting_actions(grounded_effect, executing_actions)
        if len(interacting_actions) == 1:
            self.add_must_be_effect_to_cnf(interacting_actions[0], {grounded_effect}, literals_cnf)
            return

        concurrent_execution = []
        for interacting_action in interacting_actions:
            lifted_match = self._lift_predicate(grounded_effect, interacting_action)
            concurrent_execution.append((interacting_action.name, lifted_match.untyped_representation))

        literals_cnf[grounded_effect.lifted_untyped_representation].add_possible_effect(concurrent_execution)

    def add_must_be_effect_to_cnf(self, executed_action: ActionCall,
                                  grounded_effects: Set[GroundedPredicate],
                                  literals_cnf: Dict[str, LiteralCNF]) -> NoReturn:
        """

        :param executed_action:
        :param grounded_effects:
        :param literals_cnf:
        :return:
        """
        for grounded_literal in grounded_effects:
            lifted_effect = self._lift_predicate(grounded_literal, executed_action)
            if lifted_effect is None:
                continue

            literals_cnf[grounded_literal.lifted_untyped_representation].add_possible_effect(
                [(executed_action.name, lifted_effect.untyped_representation)])

    def update_single_agent_executed_action(self, executed_action: ActionCall, previous_state: State, next_state: State,
                                            observed_objects: Dict[str, PDDLObject]) -> NoReturn:
        """

        :param executed_action:
        :param previous_state:
        :param next_state:
        :param observed_objects:
        :return:
        """
        positive_next_state_predicates, negative_next_state_predicates = \
            self.create_fully_observable_predicates(next_state, observed_objects)
        observed_action = self.partial_domain.actions[executed_action.name]
        if executed_action.name not in self.observed_actions:
            super()._add_new_action_preconditions(executed_action, observed_action, observed_objects, previous_state)
            self.observed_actions.append(observed_action.name)

        else:
            super()._update_action_preconditions(executed_action, previous_state)

        self.add_not_effect_to_cnf(executed_action, negative_next_state_predicates, self.positive_literals_cnf)
        self.add_not_effect_to_cnf(executed_action, positive_next_state_predicates, self.negative_literals_cnf)
        grounded_add_effects, grounded_del_effects = extract_effects(previous_state, next_state)
        self.add_must_be_effect_to_cnf(executed_action, grounded_add_effects, self.positive_literals_cnf)
        self.add_must_be_effect_to_cnf(executed_action, grounded_del_effects, self.negative_literals_cnf)

    def update_multiple_executed_actions(
            self, joint_action: JointActionCall, previous_state: State, next_state: State,
            observed_objects: Dict[str, PDDLObject]) -> NoReturn:
        """

        :param joint_action:
        :param previous_state:
        :param next_state:
        :param observed_objects:
        :return:
        """
        executing_actions = self._locate_executing_action(joint_action)
        for executed_action in executing_actions:
            observed_action = self.partial_domain.actions[executed_action.name]
            if executed_action.name not in self.observed_actions:
                super()._add_new_action_preconditions(executed_action, observed_action, observed_objects,
                                                      previous_state)
                self.observed_actions.append(observed_action.name)

            else:
                super()._update_action_preconditions(executed_action, previous_state)

        positive_next_state_predicates, negative_next_state_predicates = \
            self.create_fully_observable_predicates(next_state, observed_objects)
        grounded_add_effects, grounded_del_effects = extract_effects(previous_state, next_state)
        for executed_action in executing_actions:
            self.add_not_effect_to_cnf(executed_action, negative_next_state_predicates, self.positive_literals_cnf)
            self.add_not_effect_to_cnf(executed_action, positive_next_state_predicates, self.negative_literals_cnf)

        for grounded_add_effect in grounded_add_effects:
            self.handle_concurrent_execution(grounded_add_effect, executing_actions, self.positive_literals_cnf)

        for grounded_del_effect in grounded_del_effects:
            self.handle_concurrent_execution(grounded_del_effect, executing_actions, self.negative_literals_cnf)

    def handle_multi_agent_trajectory_component(
            self, component: MultiAgentComponent, objects: Dict[str, PDDLObject]) -> NoReturn:
        """Handles a single multi agent triplet in the observed trajectory.

        :param component: the triplet to handle.
        :param objects: the objects that were observed in the trajectory.
        :param agents: the names of the agents that participate in the actions.
        """
        previous_state = component.previous_state
        joint_action = component.grounded_joint_action
        next_state = component.next_state

        action_count = joint_action.action_count
        if action_count == 1:
            executing_action = self._locate_executing_action(joint_action)[0]
            self.update_single_agent_executed_action(executing_action, previous_state, next_state, objects)
            return

        self.logger.debug("More than one action is being executed in the current triplet.")
        self.update_multiple_executed_actions(joint_action, previous_state, next_state, objects)

    def learn_combined_action_model(
            self, observations: List[MultiAgentObservation]) -> Tuple[LearnerDomain, Dict[str, str]]:
        """Learn the SAFE action model from the input multi-agent trajectories.

        :param observations: the multi-agent observations.
        :return: a domain containing the actions that were learned.
        """
        self.logger.info("Starting to learn the action model!")
        self._initialize_cnfs()
        for observation in observations:
            for component in observation.components:
                self.handle_multi_agent_trajectory_component(component, observation.grounded_objects)

        self.construct_safe_actions()
        self.logger.info("Finished learning the action model!")
        learning_report = {action_name: "OK" for action_name in self.partial_domain.actions}
        return self.partial_domain, learning_report
