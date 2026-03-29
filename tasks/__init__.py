"""Aegis Server -- graders and task registry package."""

from .grader_bonus import grade_bonus
from .grader_easy import grade_easy
from .grader_hard import grade_hard
from .grader_medium import grade_medium
from .task_registry import ALL_TASKS, get_task, list_tasks

__all__ = [
    "grade_easy",
    "grade_medium",
    "grade_hard",
    "grade_bonus",
    "ALL_TASKS",
    "get_task",
    "list_tasks",
]
