"""Text-completion adapter for the tools-free conversational ablation."""

from __future__ import annotations

from .runtime import NativeToolAgentBackend
from ..llm import LLMToolCall
from ..operators import PatchEngine, apply_rewrite, parse_file_blocks
from ..workspace import FileWorkspace, GitWorkspace
from .contracts import AgentBackend, AgentSessionRequest, AgentSessionResult
from .tools import AgentToolContext, AgentToolError
from .tools import WorkspaceWriteTool, resolve_workspace_path


class ConversationalAgentBackend:
    """Apply a tools-free runtime's final text to its current workspace.

    Conversation, accounting, limits, feedback, and session history remain
    owned by the wrapped backend. This adapter only materializes the same
    response formats used by SingleShotProposer.
    """

    def __init__(self, backend: NativeToolAgentBackend , *, language: str = "python"):
        if not isinstance(backend, AgentBackend):
            raise TypeError("backend must implement AgentBackend")
        if not isinstance(language, str) or not language.strip():
            raise ValueError("language must be non-empty")
        self.backend = backend
        self.language = language
        self._patch_engine = PatchEngine()
        self._writer = WorkspaceWriteTool()

    def run(self, request: AgentSessionRequest) -> AgentSessionResult:
        result = self.backend.run(request)
        if not isinstance(result, AgentSessionResult):
            return result
        if result.final_message.strip():
            try:
                self._apply_completion(request, result.final_message)
            except (AgentToolError, OSError, TypeError, UnicodeError):
                # Malformed output remains a normal no-change preflight
                # failure, so the same session receives structured feedback.
                pass
        return result

    def release(self, session_id: str) -> None:
        self.backend.release(session_id)

    def _apply_completion(
        self,
        request: AgentSessionRequest,
        completion: str,
    ) -> None:
        workspace = request.parent.workspace
        main_path = self._main_path(workspace)
        current = (request.workdir / main_path).read_text(encoding="utf-8")

        edits = (
            parse_file_blocks(completion)
            if request.operator != "revise"
            else {}
        )
        if not edits:
            outcome = (
                self._patch_engine.apply(current, completion)
                if request.operator == "revise"
                else apply_rewrite(current, completion, self.language)
            )
            if not outcome.ok:
                return
            assert outcome.new_code is not None
            edits = {main_path: outcome.new_code}

        # Validate every path before the first write so malformed multi-file
        # output cannot leave a partially applied conversational proposal.
        for path, content in edits.items():
            resolved = resolve_workspace_path(request.workdir, path)
            if resolved.exists() and not resolved.is_file():
                return
            if len(content.encode("utf-8")) > self._writer.max_write_bytes:
                return

        context = AgentToolContext(
            workdir=request.workdir,
            parent=request.parent,
            operator=request.operator,
            preflight=request.preflight,
            remaining_timeout_s=request.limits.timeout_s,
        )
        for index, (path, content) in enumerate(edits.items()):
            self._writer.invoke(
                LLMToolCall(
                    call_id=f"completion-write-{index}",
                    name=self._writer.definition.name,
                    arguments={"path": path, "content": content},
                ),
                context,
            )

    @staticmethod
    def _main_path(workspace: FileWorkspace | GitWorkspace) -> str:
        if isinstance(workspace, FileWorkspace):
            return workspace.filename
        if isinstance(workspace, GitWorkspace):
            return workspace.main_file
        raise TypeError("unsupported workspace implementation")
