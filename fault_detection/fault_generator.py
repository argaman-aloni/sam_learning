"""Module that generates faults in a domain."""
import logging
import random
from pathlib import Path
from typing import NoReturn, Optional

from anytree import AnyNode
from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain, Action, NumericalExpressionTree

from fault_detection.defect_types import DefectType
from sam_learning.core import ConditionType, LearnerDomain


class FaultGenerator:
    """Class that generates faults in a domain."""

    working_directory_path: Path
    model_domain_file_name: str
    model_domain_file_path: Path
    logger: logging.Logger

    def __init__(self, work_dir_path: Path, model_domain_file_name: str):
        self.working_directory_path = work_dir_path
        self.model_domain_file_name = model_domain_file_name
        self.model_domain_file_path = self.working_directory_path / self.model_domain_file_name
        self.logger = logging.getLogger(__name__)

    def alter_action_numeric_precondition(self, faulty_action: Action) -> NoReturn:
        """Alters the action's preconditions so that it will contain a defect.

        :param faulty_action: the action to alter.
        """
        self.logger.info(f"Altering the action - {faulty_action.name} preconditions!")
        precondition_to_alter: NumericalExpressionTree = random.choice(list(faulty_action.numeric_preconditions))
        self.logger.debug(f"Precondition to alter: {precondition_to_alter.to_pddl()}")
        if precondition_to_alter.root.value == ">=" or precondition_to_alter.root.value == ">":
            precondition_to_alter.root.value = "<="
            self.logger.debug(f"Altered precondition: {precondition_to_alter.to_pddl()}")
            return

        if precondition_to_alter.root.value == "<=" or precondition_to_alter.root.value == "<":
            precondition_to_alter.root.value = ">="
            self.logger.debug(f"Altered precondition: {precondition_to_alter.to_pddl()}")
            return

    @staticmethod
    def _alter_numeric_expression(expression: NumericalExpressionTree, should_decrease: bool = False) -> NoReturn:
        """Alters a numeric expression.

        :param expression: the expression to alter.
        :param should_decrease: whether the expression should be decreased or increased.
        """
        node = expression.root
        alter_by = random.randint(1, 20)
        while not node.is_leaf:
            node = node.children[1]

        if isinstance(node.value, float):
            node.value = node.value + alter_by
            node.id = str(node.value)

        else:
            function_name = node.id
            function_value = node.value
            new_add_node = AnyNode(
                id="-" if should_decrease else "+",
                value="-" if should_decrease else "+",
                children=[
                    AnyNode(id=function_name, value=function_value),
                    AnyNode(id=str(alter_by), value=alter_by)
                ])
            left_sibling = node.parent.children[0]
            node.parent.children = (left_sibling, new_add_node)

    def alter_action_numeric_precondition_value(self, faulty_action: Action) -> NoReturn:
        """Alter the action's effects so that it will contain a defect.

        :param faulty_action: the action to alter.
        """
        self.logger.info(f"Altering the action - {faulty_action.name} numeric precondition value!")
        precondition_to_alter: NumericalExpressionTree = random.choice(list(faulty_action.numeric_preconditions))
        root_node = precondition_to_alter.root
        should_decrease = root_node.value == ">="
        self._alter_numeric_expression(precondition_to_alter, should_decrease)
        self.logger.debug(f"Altered precondition: {precondition_to_alter.to_pddl()}")

    def alter_action_numeric_effect(self, faulty_action: Action) -> NoReturn:
        """Alter the action's effects so that it will contain a defect.

        :param faulty_action: the action to alter.
        """
        self.logger.info(f"Altering the action - {faulty_action.name} effects!")
        effect_to_alter: NumericalExpressionTree = random.choice(list(faulty_action.numeric_effects))
        self._alter_numeric_expression(effect_to_alter)
        self.logger.debug(f"Altered effect: {effect_to_alter.to_pddl()}")

    def remove_predicate_from_action(self, faulty_action: Action) -> NoReturn:
        """Remove a predicate from the action's precondition.

        :param faulty_action: the action to remove the predicate from.
        """
        self.logger.info("Removing a predicate from the action's precondition!")
        faulty_action.positive_preconditions.remove(random.choice(list(faulty_action.positive_preconditions)))

    def remove_numeric_precondition(self, faulty_action: Action) -> NoReturn:
        """Remove a numeric precondition from the action's precondition.

        :param faulty_action: the action to remove the numeric precondition from.
        """
        self.logger.info("Removing a predicate from the action's precondition!")
        faulty_action.numeric_preconditions.remove(random.choice(list(faulty_action.numeric_preconditions)))


    @staticmethod
    def _select_action_to_alter(altered_domain: Domain) -> Action:
        """Selects an action to alter using random selection on the domain's actions.

        :param altered_domain: the domain to select the action from.
        :return: the selected action.
        """
        actions_to_choose = list(altered_domain.actions.values())
        random.shuffle(actions_to_choose)
        faulty_action = random.choice(actions_to_choose)
        return faulty_action

    def _alter_action_according_to_defect_type(self, faulty_action: Action, defect_type: DefectType) -> NoReturn:
        """Alters an action according based on a specific defect type.

        :param faulty_action: the action to add a defect to.
        :param defect_type: the type of defect to add to the action.
        """
        self.logger.debug(f"Altering the action - {faulty_action.name}!")
        if defect_type == DefectType.numeric_precondition_sign:
            self.alter_action_numeric_precondition(faulty_action)

        elif defect_type == DefectType.numeric_precondition_sign:
            self.alter_action_numeric_precondition(faulty_action)

        elif defect_type == DefectType.numeric_precondition_numeric_change:
            self.alter_action_numeric_precondition_value(faulty_action)

        elif defect_type == DefectType.numeric_effect:
            self.alter_action_numeric_effect(faulty_action)

        else:
            self.logger.debug("Removing a predicate from the action's precondition!")
            self.remove_predicate_from_action(faulty_action)

    def _alter_action_according_to_random_defect(self, faulty_action: Action) -> NoReturn:
        """Sets a random defect to an action

        :param faulty_action: the action to add a defect to.
        """
        self.logger.debug(f"Altering the action - {faulty_action.name}!")
        if len(faulty_action.numeric_preconditions) > 0:
            self.alter_action_numeric_precondition(faulty_action)

        elif len(faulty_action.numeric_effects) > 0:
            self.alter_action_numeric_effect(faulty_action)

        else:
            self.logger.debug("Removing a predicate from the action's precondition!")
            self.remove_predicate_from_action(faulty_action)

    def _set_faulty_domain_and_defected_action(
            self, defect_type: Optional[DefectType] = None, action_to_alter: Optional[str] = None) -> LearnerDomain:
        """Sets the domain fields and alters the domain's actions so that it will contain a defect.

        :return: a serializable domain object.
        """
        altered_domain = DomainParser(domain_path=self.model_domain_file_path).parse_domain()
        faulty_action = self._select_action_to_alter(altered_domain) if action_to_alter is None else \
            altered_domain.actions[action_to_alter]
        if defect_type is not None:
            self._alter_action_according_to_defect_type(faulty_action, defect_type)
        else:
            self._alter_action_according_to_random_defect(faulty_action)

        faulty_domain = LearnerDomain(altered_domain)
        for original_action, faulty_domain_action in zip(altered_domain.actions.values(),
                                                         faulty_domain.actions.values()):
            faulty_domain_action.positive_preconditions = original_action.positive_preconditions
            faulty_domain_action.inequality_preconditions = original_action.inequality_preconditions
            numeric_preconditions = [precond.to_pddl() for precond in original_action.numeric_preconditions]
            faulty_domain_action.numeric_preconditions = (numeric_preconditions, ConditionType.injunctive)
            faulty_domain_action.add_effects = original_action.add_effects
            faulty_domain_action.delete_effects = original_action.delete_effects
            numeric_effects = [effect.to_pddl() for effect in original_action.numeric_effects]
            faulty_domain_action.numeric_effects = numeric_effects

        return faulty_domain

    def generate_faulty_domain(self, defect_type: Optional[DefectType] = None,
                               action_to_alter: Optional[str] = None) -> LearnerDomain:
        """Generate d domain with a defect of sort that makes it unsafe.

        :return: the domain with the randomized defect.
        """
        faulty_domain = self._set_faulty_domain_and_defected_action(defect_type, action_to_alter)
        self.logger.debug(faulty_domain.to_pddl())
        return faulty_domain
