"""An action model learning algorithm that uses the Oblique Tree algorithm with some concepts of N-SAM Learning."""
from typing import Dict, List, Optional, Tuple, Union

from pddl_plus_parser.models import Domain, Observation

from sam_learning.core import LearnerDomain, SVMFluentsLearning
from sam_learning.core.oblique_tree_fluents_learning import ObliqueTreeFluentsLearning
from sam_learning.learners import SAMLearner


class UnsafeModelLearner(SAMLearner):
    """Class that contains the logic for learning the action model using unsafe numeric approaches."""

    learning_method: Union[ObliqueTreeFluentsLearning, SVMFluentsLearning]

    def __init__(self, partial_domain: Domain, polynomial_degree: int = 1, faulty_action_name: Optional[str] = None):
        super().__init__(partial_domain)
        self.polynom_degree = polynomial_degree
        self.faulty_action_name = faulty_action_name

    def learn_unsafe_action_model(self, positive_observations: List[Observation],
                                  negative_observations: List[Observation]) -> Tuple[LearnerDomain, Dict[str, str]]:
        """Learns the action model from the given observations using unsafe methods.

        :param positive_observations: the observations to learn the action model from.
        :param negative_observations: the negative observations containing faults in them.
        :return: the learned action model and the mapping from the action names to the action names in the
            learned domain.
        """
        self.logger.info("Learning the action model from the given observations.")
        allowed_actions = {}
        learning_metadata = {}
        super().deduce_initial_inequality_preconditions()
        for observation in positive_observations:
            for component in observation.components:
                super().handle_single_trajectory_component(component, observation.grounded_objects)

        faulty_action = self.partial_domain.actions[self.faulty_action_name]
        inequalities, condition_type = self.learning_method.learn_preconditions(positive_observations,
                                                                                negative_observations)
        if len(inequalities) > 0:
            faulty_action.numeric_preconditions = (inequalities, condition_type)

        faulty_action.numeric_effects = self.learning_method.learn_effects(positive_observations)
        allowed_actions[self.faulty_action_name] = faulty_action
        learning_metadata[self.faulty_action_name] = "OK"

        self.partial_domain.actions = allowed_actions
        return self.partial_domain, learning_metadata


class ObliqueTreeModelLearner(UnsafeModelLearner):
    """An action model learning algorithm that uses the Oblique Tree algorithm with some concepts of N-SAM Learning."""

    def __init__(self, partial_domain: Domain, polynomial_degree: int = 1, faulty_action_name: Optional[str] = None):
        super().__init__(partial_domain, polynomial_degree, faulty_action_name)
        self.learning_method = ObliqueTreeFluentsLearning(faulty_action_name, self.polynom_degree, self.partial_domain)


class SVCModelLearner(UnsafeModelLearner):
    """An action model learning algorithm that uses the Oblique Tree algorithm with some concepts of N-SAM Learning."""

    def __init__(self, partial_domain: Domain, polynomial_degree: int = 1, faulty_action_name: Optional[str] = None):
        super().__init__(partial_domain, polynomial_degree, faulty_action_name)
        self.learning_method = SVMFluentsLearning(faulty_action_name, self.polynom_degree, self.partial_domain)
