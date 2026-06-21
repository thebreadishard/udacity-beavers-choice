"""
agent_transcript.py
===================

A lightweight logging observer that records Hugging Face ``smolagents`` execution into a
simple ``.jsonl`` transcript that our own pixel-office viewer (``viewer/``) tails and
animates. We own the schema end to end: each line describes one agent state change.

The observer is attached as a ``smolagents`` *step callback*. After every agent step it
appends one JSON object per state change (thinking, tool use, tool result, done) to a
``transcript.jsonl`` file in the workspace root. Each line carries the fields the viewer
needs to animate the run; a nested ``type``/``message`` block is also included so the file
stays readable as a generic LLM-CLI-style log.

Usage
-----
Attach at agent construction::

    from agent_transcript import TranscriptObserver
    observer = TranscriptObserver(agent_id="orchestrator_agent")
    agent = ToolCallingAgent(tools=[...], model=model, step_callbacks=[observer])

...or attach to an already-built agent::

    from agent_transcript import attach_observer
    observer = attach_observer(existing_agent, agent_id="customer")

Wrap run boundaries to also emit explicit start/done markers::

    observer.log_start(task)
    result = agent.run(task)
    observer.log_done(result)

The observer never raises: any logging error is swallowed so it can never break an agent run.
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

try:  # ActionStep is only needed for post-hoc registration; import defensively.
    from smolagents.memory import ActionStep
except Exception:  # pragma: no cover - smolagents always provides this in practice
    ActionStep = None


# Map an internal state to a generic LLM-CLI-style top-level log "type" so the transcript
# also reads naturally as a conversation log (user / assistant / result).
_STATE_TO_TYPE = {
    "start": "user",
    "thinking": "assistant",
    "tool_use": "assistant",
    "tool_result": "user",
    "done": "result",
}


def _stringify(value) -> str:
    """Render any tool argument / output value as a compact, human-readable string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _message_block(state: str, content: str, tool_name: str):
    """Build a Claude-Code-style ``message`` object for the given state."""
    if state == "tool_use":
        return {"role": "assistant",
                "content": [{"type": "tool_use", "name": tool_name, "input": content}]}
    if state == "tool_result":
        return {"role": "user",
                "content": [{"type": "tool_result", "content": content}]}
    role = "user" if state == "start" else "assistant"
    return {"role": role, "content": [{"type": "text", "text": content}]}


class TranscriptObserver:
    """A ``smolagents`` step callback that appends state changes to a ``.jsonl`` transcript.

    Args:
        agent_id: Identifier written to every line so the viewer can tell agents apart
            (e.g. ``"orchestrator_agent"`` vs ``"customer"``).
        transcript_path: Path to the JSONL transcript file (default ``transcript.jsonl``).
        session_id: Optional shared session id; a random one is generated if omitted.
    """

    # A process-wide lock so multiple agents writing to the same file stay line-safe.
    _file_lock = threading.Lock()

    def __init__(self, agent_id: str, transcript_path: str = "transcript.jsonl",
                 session_id: str = None):
        self.agent_id = agent_id
        self.path = Path(transcript_path)
        self.session_id = session_id or uuid4().hex

    # -- low-level emit -----------------------------------------------------------------

    def _emit(self, state: str, content: str, tool_name: str = None, step: int = None):
        """Append a single transcript line. Never raises."""
        try:
            content = content if isinstance(content, str) else _stringify(content)
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "sessionId": self.session_id,
                "uuid": uuid4().hex,
                "agent": self.agent_id,        # agent identifier (which desk/sprite)
                "state": state,                # thinking | tool_use | tool_result | done
                "type": _STATE_TO_TYPE.get(state, "assistant"),
                "content": content,            # text content
                "message": _message_block(state, content, tool_name),
            }
            if tool_name is not None:
                payload["tool_name"] = tool_name
            if step is not None:
                payload["step"] = step

            line = json.dumps(payload, ensure_ascii=False)
            with TranscriptObserver._file_lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception:
            # Logging must never break an agent run.
            pass

    # -- explicit run-boundary markers --------------------------------------------------

    def log_start(self, content: str = "") -> None:
        """Emit a 'start' event when an agent begins handling a task."""
        self._emit("start", content)

    def log_done(self, content: str = "") -> None:
        """Emit a 'done' event when an agent finishes (use around ``agent.run``)."""
        self._emit("done", content)

    # -- step callback ------------------------------------------------------------------

    def __call__(self, memory_step, agent=None) -> None:
        """smolagents step callback: translate one ActionStep into transcript lines."""
        try:
            step = getattr(memory_step, "step_number", None)
            model_output = getattr(memory_step, "model_output", None)
            tool_calls = getattr(memory_step, "tool_calls", None) or []
            observations = getattr(memory_step, "observations", None)
            is_final = bool(getattr(memory_step, "is_final_answer", False))
            action_output = getattr(memory_step, "action_output", None)

            # 1. The agent's reasoning / assistant text for this step.
            if model_output:
                self._emit("thinking", model_output, step=step)

            # 2. Any tool calls made in this step (final_answer is handled as 'done').
            final_seen = False
            for call in tool_calls:
                name = getattr(call, "name", None)
                args = getattr(call, "arguments", None)
                if name == "final_answer":
                    final_seen = True
                    self._emit("done", _stringify(args) or _stringify(action_output),
                               tool_name=name, step=step)
                else:
                    self._emit("tool_use", _stringify(args), tool_name=name, step=step)

            # 3. The observation/result text produced by the tool(s).
            if observations:
                self._emit("tool_result", observations, step=step)

            # 4. Final answer that arrived without an explicit final_answer tool call.
            if is_final and not final_seen:
                self._emit("done", _stringify(action_output), step=step)
        except Exception:
            pass


def attach_observer(agent, agent_id: str, transcript_path: str = "transcript.jsonl",
                    session_id: str = None) -> TranscriptObserver:
    """
    Attach a :class:`TranscriptObserver` to an already-constructed smolagents agent.

    Registers the observer as an ``ActionStep`` callback on the agent's callback registry
    and returns it, so the caller can also use ``log_start`` / ``log_done`` around runs.

    Args:
        agent: A smolagents agent (e.g. ``ToolCallingAgent``).
        agent_id: Identifier for this agent in the transcript.
        transcript_path: Path to the JSONL transcript file.
        session_id: Optional shared session id across agents.

    Returns:
        The attached observer instance.
    """
    observer = TranscriptObserver(agent_id, transcript_path, session_id)
    registry = getattr(agent, "step_callbacks", None)
    if registry is not None and ActionStep is not None and hasattr(registry, "register"):
        registry.register(ActionStep, observer)
    return observer
