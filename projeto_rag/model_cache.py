import os
from pathlib import Path
from typing import Optional


def is_local_model_ref(model_ref: str) -> bool:
    return Path(model_ref).expanduser().exists()


def cached_huggingface_model_path(model_ref: str) -> Optional[str]:
    if is_local_model_ref(model_ref):
        return str(Path(model_ref).expanduser())

    cache_root = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    model_dir = cache_root / "hub" / f"models--{model_ref.replace('/', '--')}"
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
    if not snapshots:
        return None

    latest = max(snapshots, key=lambda path: path.stat().st_mtime)
    return str(latest)


def resolve_model_ref(model_ref: str, offline: bool) -> Optional[str]:
    if not offline:
        return model_ref
    return cached_huggingface_model_path(model_ref)
