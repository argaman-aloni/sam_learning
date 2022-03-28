"""Module for exceptions that will be used in the algorithms."""


class NotSafeActionError(Exception):
    action_name: str
    reason: str

    def __init__(self, name: str, reason: str):
        self.action_name = name
        self.reason = reason

    def __str__(self):
        return f"The action {self.action_name} is not safe to use! The reason - {self.reason}"
