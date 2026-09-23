"""A goal as the verifier knows it: its key, its environment, its name.

`Goal.identity` is the local memo key and stays that way. The verifier's key is
a different thing -- bound to one environment, exact where the memo key is
coarse -- and merging the two into one field would make one field answer to two
contracts. So the verifier's side lives beside the node, in `goal_contracts`,
written once when the goal is first resolved and read back by every request
made about it.

What a request needs, all of it here:

- the goal key and the environment (`base`) it was resolved in;
- the proposition that was resolved, and the context and options it was
  resolved under -- the verifier hashes all three into the key, and publishes
  only when the context matches the one on record;
- the namespace prefix the verifier saw, because it reports roots fully
  qualified and a root built from the short name would not match.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .run_solver import declaration_name

_OPEN = "([{⦃"
_CLOSE = ")]}⦄"


class StatementError(ValueError):
    """A goal statement this side cannot turn into a proposition by text."""


def proposition_of(statement: str) -> str:
    """`theorem foo (a b : Nat) (h : P a) : Q a b` -> `∀ (a b : Nat) (h : P a), Q a b`.

    The verifier resolves a proposition, not a declaration. A theorem's type is
    the Pi type over its binders, so quantifying the binders over the
    conclusion gives the same term -- and the verifier's key ignores binder
    names, so the exact spelling does not matter. Binder kinds do: `{a}` stays
    implicit, `[inst]` stays an instance binder.

    Split at the first `:` at bracket depth zero after the name. `:=` never
    appears in a signature, so a colon there is the type ascription.
    """

    body = statement.strip()
    for keyword in ("theorem", "lemma"):
        if body.startswith(keyword + " "):
            body = body[len(keyword) + 1:].lstrip()
            break
    else:
        raise StatementError(f"not a theorem or lemma: {statement[:80]!r}")
    name = declaration_name("theorem " + body)
    rest = body[len(name):]
    depth = 0
    for index, char in enumerate(rest):
        if char in _OPEN:
            depth += 1
        elif char in _CLOSE:
            depth -= 1
        elif char == ":" and depth == 0:
            if rest[index + 1:index + 2] == "=":
                break
            binders = rest[:index].strip()
            conclusion = rest[index + 1:].strip()
            if not conclusion:
                break
            return f"∀ {binders}, {conclusion}" if binders else conclusion
    raise StatementError(f"no type ascription in {statement[:80]!r}")


@dataclass(frozen=True)
class GoalContract:
    """The verifier's record of one goal, as this side keeps it."""

    goal_key: str
    #: The full key object, exactly as the verifier returned it. Requests send
    #: it back verbatim; nothing here recomputes it.
    goal_key_obj: Mapping[str, Any]
    base: Mapping[str, Any]
    name_prefix: str
    proposition: str
    context: Mapping[str, Any] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)

    def qualify(self, short_name: str) -> str:
        return self.name_prefix + short_name

    def to_row(self) -> dict[str, Any]:
        return {
            "goal_key": self.goal_key,
            "goal_key_json": _canonical(self.goal_key_obj),
            "base_json": _canonical(self.base),
            "name_prefix": self.name_prefix,
            "proposition": self.proposition,
            "context_json": _canonical(self.context),
            "options_json": _canonical(self.options),
        }

    @classmethod
    def from_row(cls, row) -> "GoalContract":
        return cls(
            goal_key=row["goal_key"],
            goal_key_obj=json.loads(row["goal_key_json"]),
            base=json.loads(row["base_json"]),
            name_prefix=row["name_prefix"],
            proposition=row["proposition"],
            context=json.loads(row["context_json"]),
            options=json.loads(row["options_json"]),
        )

    @classmethod
    def from_resolved(
        cls,
        resolved: Mapping[str, Any],
        *,
        base: Mapping[str, Any],
        proposition: str,
        context: Mapping[str, Any],
        options: Mapping[str, Any] | None = None,
    ) -> "GoalContract":
        """From the verifier's `ResolvedGoal` answer."""

        key = resolved["goal_key"]
        return cls(
            goal_key=key["key"],
            goal_key_obj=dict(key),
            base=dict(base),
            name_prefix=resolved["name_prefix"],
            proposition=proposition,
            context=dict(context),
            options=dict(options or {}),
        )


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = ["GoalContract", "StatementError", "proposition_of"]
