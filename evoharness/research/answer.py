"""Sign a governance card, from a terminal.

The durable signing path, and for now the only one. `evoweb` is gone and the
session tools are read-only by construction, so the queue has been accepting
cards with nothing able to take them out.

It is written before DH-5's session signing rather than after, because the
plan makes the order load-bearing: a session approval blocks one tool call
while these decisions span days, so signing from a session can only ever be
opportunistic — somebody happening to have one open. A path that only works
when a session is open cannot be the only path, and the way to guarantee that
is for the other one to exist first.

Two rules here are the same as the ones DH-5's third layer will need, for the
same reason.

The actor is bound by this process, never passed in. `InboxStore` already
says a card body may not nominate its own approver; an `--actor` flag is that
same defect wearing argv, since whoever runs the command would choose the
name that lands in the audit record. The operating-system user is this
interface's authentication context, so that is what signs.

Answering is irreversible — the inbox refuses a second answer to one card —
so the card is shown and its id has to be typed back before anything is
written. There is deliberately no flag to skip that. A caller who wants no
confirmation wants `InboxStore.answer` directly, and having to say so in code
is the friction, not an obstacle to route around.
"""

from __future__ import annotations

import getpass
from pathlib import Path

from .inbox import DecisionRequest, InboxStore
from .store import ResearchStore

#: Who may sign, one identity per line, beside the ledger it governs. A file
#: rather than a flag or an environment variable: both of those are chosen by
#: the caller at the moment of signing, and that is precisely the choice this
#: must not leave to them.
ACTORS_FILE = "authorized_actors.txt"

LEDGER_NAME = "research.sqlite3"

#: Recorded on every decision this module produces, so an audit can tell a
#: signature made here from one made through some future interface.
SOURCE = "cli"


class AnswerRefused(RuntimeError):
    """Nothing was recorded, and the reason is the operator's to act on."""


def authorized_actors(research_root: Path | str) -> frozenset[str]:
    """Read the identities permitted to sign against this ledger."""

    path = Path(research_root) / ACTORS_FILE
    if not path.is_file():
        raise AnswerRefused(
            f"no {ACTORS_FILE} beside the ledger at {research_root}. Create it "
            "with one identity per line; it decides who may sign, which is why "
            "it is not something this command can be told."
        )
    actors = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not actors:
        raise AnswerRefused(f"{path} lists no identities")
    return frozenset(actors)


def open_inbox(research_root: Path | str) -> InboxStore:
    """Open the ledger for writing, refusing to invent one.

    `ResearchStore` runs its schema DDL on construction, so a mistyped root
    would manufacture an empty ledger and then report an empty queue — the
    same trap `readout.governance` avoids on the read side. The existence
    check has to come first.
    """

    ledger = Path(research_root) / LEDGER_NAME
    if not ledger.is_file():
        raise AnswerRefused(f"no research ledger at {ledger}")
    return InboxStore(
        ResearchStore(Path(research_root)),
        authorized_actors=authorized_actors(research_root),
    )


def render(request: DecisionRequest) -> str:
    """The card, put in front of the person about to sign it.

    `default` is included because it is what happens if they walk away, and a
    signer who does not know that cannot tell approving from doing nothing.
    """

    lines = [
        f"  request   {request.request_id}",
        f"  kind      {request.kind}",
        f"  subject   {request.experiment_id}",
        f"  question  {request.question}",
        f"  allowed   {', '.join(request.allowed_actions)}",
        f"  default   {request.default_action}   (if nobody acts)",
    ]
    if request.consequence_of_waiting:
        lines.append(f"  waiting   {request.consequence_of_waiting}")
    if request.uncertainty:
        lines.append(f"  unknown   {request.uncertainty}")
    if request.evidence_refs:
        lines.append(f"  evidence  {len(request.evidence_refs)} reference(s)")
    return "\n".join(lines)


def retype_the_id(request_id: str, read=input) -> bool:
    """Confirm by producing the identifier rather than accepting a suggestion.

    A yes/no prompt is answered by reflex. Retyping the id is the cheapest
    thing that cannot be done without having looked at what is on screen.
    """

    answer = read("type the request id to sign, anything else to abort\n> ")
    return answer.strip() == request_id


def current_actor() -> str:
    return getpass.getuser()


def answer_card(
    research_root: Path | str,
    request_id: str,
    *,
    action: str,
    reason: str,
    actor: str | None = None,
    confirm=retype_the_id,
    show=print,
) -> dict:
    """Show one card, confirm, and record the decision.

    :param actor: overridable for tests only. Production passes nothing and
        gets the operating-system user; see the module docstring for why this
        is not a command-line flag.
    """

    if not reason.strip():
        raise AnswerRefused(
            "a reason is required: a decision nobody explained cannot be "
            "reviewed later"
        )

    signer = current_actor() if actor is None else actor
    inbox = open_inbox(research_root)
    request = inbox.get(request_id)

    show(render(request))
    show(f"\nsigning as {signer}: {action}")
    if not confirm(request_id):
        raise AnswerRefused("aborted; nothing was recorded")

    decision = inbox.answer(
        request_id,
        action=action,
        actor=signer,
        reason=reason,
        source=SOURCE,
    )
    return decision.to_json()
