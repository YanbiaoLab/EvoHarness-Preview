"""Read a run directory from outside Python.

The library decides what a run directory means; this only picks which view to
print. A caller in another language that parsed `run.db` and `manifest.json`
itself would be reimplementing the parts that are easy to get wrong — the
in-flight sentinel, the manifest-over-checkpoint precedence, the stall verdict
— and the two would drift apart without either side noticing.

Output is JSON on stdout in every case, failures included, so a caller has one
parse path. Which case it was is the exit code.
"""

from __future__ import annotations

import argparse
import json
import sys

from .detail import candidate_detail, trajectory
from .governance import card, pending_cards, recent_decisions
from .peer import peer_view
from .status import ReadoutError, list_runs, run_status

EXIT_OK = 0
#: Something nobody anticipated. Distinct from EXIT_REFUSED so a caller can
#: tell "you asked for the wrong thing" from "this is a bug".
EXIT_UNEXPECTED = 1
EXIT_REFUSED = 2

def _emit(payload: object) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2, default=str)
    sys.stdout.write("\n")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog= "python -m evoharness.readout", description=__doc__
    )
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="one run's state, cheaply")
    status.add_argument("--run-dir", required=True)

    listing = sub.add_parser("list", help="every run under a root")
    listing.add_argument("--root", required=True)

    traj = sub.add_parser("trajectory", help="generation by generation")
    traj.add_argument("--run-dir", required=True)
    traj.add_argument(
        "--detail",
        choices=("summary", "full"),
        default="summary",
        help="summary counts each generation; full lists every candidate",
    )

    cand = sub.add_parser("candidate", help="one candidate, with its program")
    cand.add_argument("--run-dir", required=True)
    cand.add_argument("--id", required=True)

    peer = sub.add_parser(
        "peer",
        help="one candidate as another candidate may see it (narrow view)",
    )
    peer.add_argument("--run-dir", required=True)
    peer.add_argument("--id", required=True)
    peer.add_argument(
        "--path",
        default=None,
        help="a file inside that candidate; omit for its inventory",
    )

    # Governance. Read-only by construction: there is no `answer` subcommand
    # here and there is no function behind one — a card is signed through the
    # path that binds an actor to an authenticated identity, not through a
    # view.
    cards = sub.add_parser("cards", help="decision requests awaiting a person")
    cards.add_argument("--research-root", required=True)

    one_card = sub.add_parser("card", help="one decision request in full")
    one_card.add_argument("--research-root", required=True)
    one_card.add_argument("--id", required=True)

    decided = sub.add_parser("decided", help="recently answered cards")
    decided.add_argument("--research-root", required=True)
    decided.add_argument("--limit", type=int, default=20)

    return parser


def _dispatch(args: argparse.Namespace) -> object:
    if args.command == "status":
        return run_status(args.run_dir).to_json()
    elif args.command == "list":
        return [status.to_json() for status in list_runs(args.root)]
    elif args.command == "trajectory":
        rows = trajectory(args.run_dir)
        if args.detail == "full":
            return [row.to_json() for row in rows]
        else:
            return [row.summary() for row in rows]

    elif args.command == "cards":
        return pending_cards(args.research_root)
    elif args.command == "card":
        return card(args.research_root, args.id)
    elif args.command == "decided":
        return recent_decisions(args.research_root, args.limit)
    elif args.command == "peer":
        # Not candidate_detail with fewer keys — a separate view, so a field
        # added to EvalReport later cannot reach a candidate by default.
        return peer_view(args.run_dir, args.id, args.path)

    detail = candidate_detail(args.run_dir, args.id)
    if detail is None:
        # Not yet written and never existed look the same from here, so this
        # is a refusal rather than a null: a caller polling for a candidate
        # would otherwise read "null" as "the run has no candidates".
        raise ReadoutError(f"no candidate {args.id!r} in {args.run_dir}")

    return detail


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _emit(_dispatch(args))
    except ReadoutError as exc:
        _emit({"error": str(exc), "kind": "readout"})
        return EXIT_REFUSED
    except Exception as exc:  # noqa: BLE001
        # Deliberately broad, because this is a process boundary. A traceback
        # on stdout is unparseable to the caller, and one on stderr with a
        # bare exit code is indistinguishable from the interpreter failing to
        # start at all.
        _emit({"error": f"{type(exc).__name__}: {exc}", "kind": "unexpected"})
        return EXIT_UNEXPECTED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
