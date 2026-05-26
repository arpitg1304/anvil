"""Per-episode sidecar writer for ``anvil guard``.

When ``guard`` wraps a recording command with a LeRobot-style dataset
directory, this module identifies the episodes recorded in this session
(by snapshotting the dataset directory before vs. after the record
command runs) and writes an ``EpisodeReport`` JSON next to each one.

Layout, per [anvil_plan.md §8](../../anvil_plan.md):

    <dataset>/
    ├── data/chunk-000/file_000.parquet
    ├── videos/chunk-000/episode_000042/cam_high.mp4
    ├── meta/
    └── anvil/                              # this module's territory
        ├── manifest_hash.txt               # pin reference, session-wide
        ├── episode_000042.anvil.json       # one per recorded episode
        └── episode_000043.anvil.json

The sidecar is a schema-compliant ``EpisodeReport`` JSON. Forge (or any
other tool) can ingest it without depending on the anvil Python package
— the contract is the JSON file and the schema in
``docs/metadata_schema.md``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from anvil.schema import EpisodeReport

ANVIL_SIDECAR_SUBDIR = "anvil"
MANIFEST_HASH_FILENAME = "manifest_hash.txt"
_EPISODE_DIR_PATTERN = re.compile(r"^episode_\d+$")


def anvil_dir(dataset_dir: Path) -> Path:
    """Where sidecars and the session manifest hash live."""
    return dataset_dir / ANVIL_SIDECAR_SUBDIR


def sidecar_path(dataset_dir: Path, episode_id: str) -> Path:
    return anvil_dir(dataset_dir) / f"{episode_id}.anvil.json"


def manifest_hash_path(dataset_dir: Path) -> Path:
    return anvil_dir(dataset_dir) / MANIFEST_HASH_FILENAME


def snapshot_episodes(dataset_dir: Path) -> set[str]:
    """Return existing episode IDs in a LeRobot-style dataset.

    Inspects ``<dataset>/videos/*/episode_NNNNNN/`` per the v3 layout.
    Returns an empty set if either the dataset directory or ``videos/``
    subdir is missing (fresh dataset).
    """
    if not dataset_dir.exists():
        return set()
    videos_dir = dataset_dir / "videos"
    if not videos_dir.exists():
        return set()
    out: set[str] = set()
    for ep_dir in videos_dir.glob("*/episode_*"):
        if ep_dir.is_dir() and _EPISODE_DIR_PATTERN.match(ep_dir.name):
            out.add(ep_dir.name)
    return out


def write_sidecars(
    dataset_dir: Path,
    episode_ids: Iterable[str],
    base_report: EpisodeReport,
) -> list[Path]:
    """Write one sidecar per episode id, using ``base_report`` as the template.

    Each sidecar is a copy of ``base_report`` with its ``episode_id`` field
    overwritten to match the directory it describes. Returns the paths
    written in sorted order.
    """
    out_dir = anvil_dir(dataset_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ep_id in sorted(episode_ids):
        sidecar = base_report.model_copy(update={"episode_id": ep_id})
        path = sidecar_path(dataset_dir, ep_id)
        path.write_text(sidecar.model_dump_json(indent=2), encoding="utf-8")
        written.append(path)
    return written


def write_session_manifest_hash(dataset_dir: Path, manifest_hash: str) -> Path:
    """Write the session-wide ``manifest_hash.txt`` pointing at the pin used."""
    out_dir = anvil_dir(dataset_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_hash_path(dataset_dir)
    path.write_text(manifest_hash + "\n", encoding="utf-8")
    return path


__all__ = [
    "ANVIL_SIDECAR_SUBDIR",
    "MANIFEST_HASH_FILENAME",
    "anvil_dir",
    "manifest_hash_path",
    "sidecar_path",
    "snapshot_episodes",
    "write_session_manifest_hash",
    "write_sidecars",
]
