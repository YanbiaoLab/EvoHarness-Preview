from evoharness.evocore.sanitize import AllowlistSanitizer


def _san():
    return AllowlistSanitizer(
        item_allow={"optimizer": frozenset({"item_id", "passed", "proof"})},
        summary_allow={"optimizer": frozenset({"score"})},
    )


def test_allowlist_filters_and_keeps_only_allowed():
    item = {"item_id": "P1", "passed": False, "proof": "ok", "secret": "参考解"}
    out = _san().sanitize_item(item, "optimizer")
    assert out == {"item_id": "P1", "passed": False, "proof": "ok"}
    assert "secret" not in out


def test_unknown_audience_is_fail_closed():
    item = {"item_id": "P1", "proof": "ok"}
    assert _san().sanitize_item(item, "nobody") == {}
    assert _san().sanitize_summary({"score": 1}, "nobody") == {}
