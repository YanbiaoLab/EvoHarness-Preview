"""Answer a governance card from a terminal.

The write side of the research layer's human interface. Reading lives in
`evoharness.readout`, which opens the ledger read-only and has no answer path
to forget to guard; this is the one place that signs, and it is separate for
that reason.

There is no `--actor`. The identity comes from the operating-system user and
is checked against a file beside the ledger — see `answer.py` for why an
identity the caller supplies at signing time is no identity at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .answer import AnswerRefused, answer_card
from .inbox import InboxError

EXIT_OK = 0
EXIT_UNEXPECTED = 1
EXIT_REFUSED = 2


def _emit(payload: object) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2, default=str)
    sys.stdout.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evoharness.research", description=__doc__
    )
    sub = parser.add_subparsers(dest="command", required=True)

    signer = sub.add_parser("answer", help="sign one decision card")
    signer.add_argument("--research-root", type=Path, required=True)
    signer.add_argument("--id", required=True, help="the request id to answer")
    signer.add_argument(
        "--action",
        required=True,
        help="one of the card's allowed actions; the card lists them",
    )
    signer.add_argument(
        "--reason",
        required=True,
        help="why. Recorded verbatim in the audit ledger, and not optional: "
        "a decision nobody explained cannot be reviewed later.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        recorded = answer_card(
            args.research_root,
            args.id,
            action=args.action,
            reason=args.reason,
        )
    except (AnswerRefused, InboxError) as exc:
        _emit({"error": str(exc), "kind": type(exc).__name__})
        return EXIT_REFUSED
    except Exception as exc:  # noqa: BLE001 — process boundary
        _emit({"error": f"{type(exc).__name__}: {exc}", "kind": "unexpected"})
        return EXIT_UNEXPECTED
    _emit(recorded)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
