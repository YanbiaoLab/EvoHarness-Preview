"""Authored tasks: a directory of files becomes a runnable task.

The scoring rule stays ordinary Python owned by the domain expert. What lives
here is only the mechanical part — read the declaration, refuse the malformed
ones, and give the task an identity that is its contents rather than its path.
"""

from .command_check import DEFAULT_TIMEOUT_S, CommandCheck
from .task_dir import (
    TASK_FILE,
    TaskDirError,
    directory_content_hash,
    load_task_from_dir,
)

__all__ = [
    "DEFAULT_TIMEOUT_S",
    "CommandCheck",
    "TASK_FILE",
    "TaskDirError",
    "directory_content_hash",
    "load_task_from_dir",
]
