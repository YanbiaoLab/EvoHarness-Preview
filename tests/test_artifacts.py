import pytest

from evoharness.evocore.artifacts import ArtifactRef, FileArtifactStore


def test_put_open_layered_access(tmp_path):
    store = FileArtifactStore(tmp_path / "artifacts")
    ref = store.put("cand-1", {
        "summary": {"n": 2, "n_correct": 1},
        "items": [
            {"item_id": "P1", "passed": True, "proof": "ok"},
            {"item_id": "P2", "passed": False, "proof": "bad", "critique": "缺论证"},
        ],
    })
    assert ref.encode() == "file:cand-1"

    view = store.open(ref)
    assert view.summary() == {"n": 2, "n_correct": 1}
    assert set(view.item_ids()) == {"P1", "P2"}
    assert view.item_ids(failed_only=True) == ["P2"]
    assert view.item("P2")["critique"] == "缺论证"
    assert view.search("缺论证") == ["P2"]


def test_ref_roundtrip_and_path_safety(tmp_path):
    assert ArtifactRef.decode(ArtifactRef("file", "c1").encode()) == ArtifactRef("file", "c1")
    store = FileArtifactStore(tmp_path)
    with pytest.raises(ValueError):
        store.put("../escape", {"items": []})
