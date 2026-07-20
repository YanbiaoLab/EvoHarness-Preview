"""Signature recording is always-on in every recipe (E0 included) so that
populations from all groups carry behavior signatures and stay comparable in
post-hoc diversity analysis. Only the *intervention* (dedup + down-weighting)
is group-dependent."""

from evoharness.evoplus import SignatureRecorder

__all__ = ["SignatureRecorder"]
