import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class ArtifactRef:
    store: str
    key: str

    def encode(self) -> str:
        return f"{self.store}:{self.key}"

    @classmethod
    def decode(cls, s: str) -> "ArtifactRef":
        store, _, key = s.partition(":")

        if not store or not key:
            raise ValueError(f"invalid artifact ref: {s!r}")
        return cls(store, key)

@runtime_checkable
class TraceView(Protocol):
    def summary(self) -> dict: ...

    def item_ids(self, *, failed_only: bool = False) -> list[str]: ...

    def item(self, item_id: str) -> dict: ...

    def search(self, query: str) -> list[str]: ...


@runtime_checkable
class ArtifactStore(Protocol):
    def put(self, candidate_id: str, blob: dict) -> ArtifactRef: ...
    def open(self, ref: ArtifactRef) -> TraceView: ...


def _safe_name(name: str) -> str:
    """禁止路径穿越:item_id / candidate_id 只能是纯文件名。"""
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError(f"unsafe artifact name: {name!r}")
    return name


class FileArtifactStore:
    store_kind = "file"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def put(self, candidate_id: str, blob: dict) -> ArtifactRef:
        cand_dir = self.root / _safe_name(candidate_id)
        items_dir = cand_dir / "items"
        items_dir.mkdir(parents=True, exist_ok=True)
        item_index = []

        for item in blob.get("items", []):
            item_id = _safe_name(str(item["item_id"]))
            (items_dir / f"{item_id}.json").write_text(
                json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            item_index.append(
                {"item_id": item_id, "passed": bool(item.get("passed", False))}
            )
        (cand_dir / "index.json").write_text(
            json.dumps(
                {"summary": blob.get("summary", {}), "item_index": item_index},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return ArtifactRef(self.store_kind, candidate_id)

    def open(self, ref: ArtifactRef) -> "FileTraceView":
        if ref.store != self.store_kind:
            raise ValueError(f"ref is not a file artifact: {ref.encode()}")
        return FileTraceView(self.root / _safe_name(ref.key))



@dataclass
class FileTraceView:
    cand_dir: Path

    def _index(self) -> dict:
        return json.loads((self.cand_dir / "index.json").read_text(encoding="utf-8"))

    def summary(self) -> dict:
        return self._index().get("summary", {})

    def item_ids(self, *, failed_only: bool = False) -> list[str]:
        entries = self._index().get("item_index", [])
        if failed_only:
            entries = [e for e in entries if not e["passed"]]
        return [e["item_id"] for e in entries]

    def item(self, item_id: str) -> dict:
        path = self.cand_dir / "items" / f"{_safe_name(item_id)}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def search(self, query: str) -> list[str]:
        q = query.lower()
        hits = []
        for item_id in self.item_ids():
            blob = json.dumps(self.item(item_id), ensure_ascii=False).lower()
            if q in blob:
                hits.append(item_id)
        return hits




__all__ = [
    "ArtifactRef",
    "ArtifactStore",
    "FileArtifactStore",
    "FileTraceView",
    "TraceView",
]

