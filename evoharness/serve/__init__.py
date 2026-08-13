"""serve: reference evaluation service for the remote eval protocol
(docs/eval_protocol.md). Standalone by design: never imports core."""

from .grading import Grade, GradeContext, GradeFn, GradeValue, InfraError, coerce_grade
from .http import EvalHTTPServer, serve
from .service import PROTOCOL_VERSION, EvalService, Job

__all__ = [
    "Grade",
    "GradeContext",
    "GradeFn",
    "GradeValue",
    "InfraError",
    "coerce_grade",
    "EvalService",
    "Job",
    "PROTOCOL_VERSION",
    "EvalHTTPServer",
    "serve",
]
