"""A trickling response must not be able to consume a whole agent session.

`urlopen(timeout=)` bounds the gap between two socket reads, not the request.
Measured 2026-08-27: a single call sat for 8,141 seconds against a 400s socket
timeout and only ended when the agent session's 9,000s budget expired. The
session had completed 16 healthy turns and lost all of them.
"""

from __future__ import annotations

import pytest

from evoharness.core.llm import LLMTransientError, _read_within


class _Trickle:
    """A body that never ends and always has one more byte ready.

    This is exactly the shape the socket timeout cannot catch: every read
    succeeds, so the socket's clock is reset every time.
    """

    def __init__(self, clock):
        self.clock = clock
        self.reads = 0

    def read(self, _n):
        self.reads += 1
        self.clock.advance(0.5)
        return b"x"


class _Clock:
    def __init__(self):
        self.now = 0.0

    def advance(self, dt):
        self.now += dt

    def __call__(self):
        return self.now


@pytest.fixture()
def clock(monkeypatch):
    c = _Clock()
    import time

    monkeypatch.setattr(time, "monotonic", c)
    return c


def test_trickling_body_is_cut_off(clock):
    body = _Trickle(clock)
    with pytest.raises(LLMTransientError) as exc:
        _read_within(body, 10.0)
    assert "wall clock" in str(exc.value)
    # Cut off near the deadline, not after thousands of reads.
    assert body.reads <= 25


def test_transient_so_the_existing_retry_path_handles_it(clock):
    """It must be *transient* — a hung call is the provider being flaky, and
    that already has bounded retries with backoff. A hard error would fail the
    proposal outright and count toward the dead-proposer breaker."""
    with pytest.raises(LLMTransientError):
        _read_within(_Trickle(clock), 5.0)


def test_normal_body_is_returned_whole(clock):
    class _Body:
        def __init__(self):
            self.parts = [b"abc", b"def", b""]

        def read(self, _n):
            return self.parts.pop(0)

    assert _read_within(_Body(), 10.0) == b"abcdef"


def test_body_that_overruns_is_discarded_even_if_it_completed(clock):
    """A body whose last chunk lands past the deadline is dropped, on purpose.

    HTTP offers no way to tell "done" from "still trickling" without issuing
    one more read, so from in here the two are indistinguishable. Keeping the
    payload would mean the ceiling only applies to responses that happen to
    stall in the middle — which is not a ceiling. A call that needed longer
    than the bound is late either way.
    """

    class _Body:
        def __init__(self):
            self.parts = [b"payload", b""]

        def read(self, _n):
            clock.advance(20.0)
            return self.parts.pop(0)

    with pytest.raises(LLMTransientError):
        _read_within(_Body(), 10.0)


def test_deadline_is_checked_before_waiting_again(clock):
    """Never start a read we already know runs past the deadline."""

    class _Body:
        def __init__(self):
            self.reads = 0

        def read(self, _n):
            self.reads += 1
            clock.advance(99.0)
            return b"x"

    body = _Body()
    with pytest.raises(LLMTransientError):
        _read_within(body, 10.0)
    assert body.reads == 1
