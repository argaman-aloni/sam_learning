from functools import singledispatch

from pddl_plus_parser.models import ObservedComponent, MultiAgentComponent


@singledispatch
def get_grounded_actions(component):
    #in case we get a not supported component type, should act like it did before this commit.
    return [component.grounded_action_call]

@get_grounded_actions.register
def _(component: ObservedComponent):
    return [component.grounded_action_call]

@get_grounded_actions.register
def _(component: MultiAgentComponent):
    return component.grounded_joint_action.operational_actions