"""Speech Commands splits that keep the official speaker groups separate.

Use a new cache and protocol for these splits. The published random-clip split
is a different experiment and must not be relabelled with this protocol.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import PurePosixPath
from typing import Iterable

import numpy as np

SPEECH_SPLIT_PROTOCOL = "speech_commands_v02_official_speakers_train_subset_v1"
PARTITIONS = ("train", "validation", "test")


def _metadata(paths: Iterable[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
    paths = list(paths)
    if len(set(paths)) != len(paths):
        raise ValueError("Speech filenames must be unique")
    labels, speakers = [], []
    for name in paths:
        path = PurePosixPath(name)
        match = re.fullmatch(r"(.+)_nohash_([0-9]+)\.wav", path.name)
        if len(path.parts) != 2 or path.is_absolute() or ".." in path.parts or not match:
            raise ValueError(
                f"Expected a relative keyword/speaker_nohash_number.wav path: {name!r}"
            )
        labels.append(path.parts[0])
        speakers.append(match.group(1))
    return paths, np.asarray(labels), np.asarray(speakers)


def _hash(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def audit_speech_split(paths: Iterable[str], indices: dict[str, np.ndarray]) -> dict:
    """Count examples, classes and speakers, including any speaker overlap."""
    paths, labels, speakers = _metadata(paths)
    parts = {}
    speaker_sets = {}
    clip_sets = {}
    for name in PARTITIONS:
        rows = np.asarray(indices[name])
        if rows.ndim != 1 or rows.dtype.kind not in "iu" or len(rows) == 0:
            raise ValueError(f"{name} must contain a nonempty integer index vector")
        if rows.min() < 0 or rows.max() >= len(paths) or len(np.unique(rows)) != len(rows):
            raise ValueError(f"{name} contains invalid or duplicate row indices")
        words, counts = np.unique(labels[rows], return_counts=True)
        speaker_sets[name] = set(speakers[rows])
        clip_sets[name] = set(map(int, rows))
        parts[name] = {
            "clips": len(rows),
            "speakers": len(speaker_sets[name]),
            "class_counts": dict(zip(words.tolist(), counts.tolist())),
            "filenames_sha256": _hash(sorted(paths[i] for i in rows)),
            "speakers_sha256": _hash(sorted(speaker_sets[name])),
        }
    overlaps = {}
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlaps[f"{first}_{second}"] = {
            "clips": len(clip_sets[first] & clip_sets[second]),
            "speakers": len(speaker_sets[first] & speaker_sets[second]),
            "second_partition_clips_from_first_partition_speakers": int(
                np.isin(speakers[indices[second]], list(speaker_sets[first])).sum()
            ),
        }
    return {"partitions": parts, "overlaps": overlaps}


def official_speech_split(
    paths: Iterable[str],
    validation_files: Iterable[str],
    testing_files: Iterable[str],
    *,
    train_size: int = 20_020,
    seed: int = 20260812,
    training_files: Iterable[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Use all official validation/test clips and a balanced training subset.

    ``paths`` contains every keyword clip, in cache row order. Official list
    entries are relative ``keyword/filename.wav`` paths. ``training_files`` can
    preserve an existing training subset exactly; otherwise each class is
    sampled without replacement using ``seed``. No clips cross speaker groups.
    """
    paths, labels, speakers = _metadata(paths)
    lookup = {name: index for index, name in enumerate(paths)}
    validation = set(validation_files)
    testing = set(testing_files)
    if not validation or not testing or validation & testing:
        raise ValueError("Official validation and testing lists must be nonempty and disjoint")
    missing = (validation | testing) - lookup.keys()
    if missing:
        raise ValueError(f"Cache is missing {len(missing)} official validation/testing clips")
    training = set(paths) - validation - testing
    pools = {"train": training, "validation": validation, "test": testing}
    official_indices = {
        name: np.asarray(sorted(lookup[path] for path in members), dtype=np.int64)
        for name, members in pools.items()
    }
    official_audit = audit_speech_split(paths, official_indices)
    if any(item["speakers"] for item in official_audit["overlaps"].values()):
        raise ValueError("Official lists split a speaker across partitions")

    classes = sorted(set(labels))
    if train_size <= 0 or train_size % len(classes):
        raise ValueError("train_size must be positive and divisible by the number of classes")
    per_class = train_size // len(classes)
    if training_files is None:
        rng = np.random.default_rng(seed)
        chosen = []
        for word in classes:
            candidates = sorted(path for path in training if path.split("/")[0] == word)
            if len(candidates) < per_class:
                raise ValueError(f"Class {word} has fewer than {per_class} official training clips")
            chosen.extend(rng.choice(candidates, per_class, replace=False).tolist())
    else:
        chosen = list(training_files)
        if (
            len(chosen) != train_size
            or len(set(chosen)) != train_size
            or not set(chosen) <= training
        ):
            raise ValueError(
                "training_files must contain train_size distinct official training clips"
            )
        if any(sum(path.split("/")[0] == word for path in chosen) != per_class for word in classes):
            raise ValueError("training_files must be class-balanced")

    indices = dict(official_indices)
    indices["train"] = np.asarray(sorted(lookup[path] for path in chosen), dtype=np.int64)
    manifest = {
        "schema_version": 1,
        "split_protocol": SPEECH_SPLIT_PROTOCOL,
        "training_selection": "supplied_filenames"
        if training_files is not None
        else "balanced_default_rng",
        "training_seed": None if training_files is not None else seed,
        "class_names": classes,
        "cache_path_order_sha256": _hash(paths),
        "official_validation_list_sha256": _hash(sorted(validation)),
        "official_testing_list_sha256": _hash(sorted(testing)),
        "full_official_partitions": official_audit,
        **audit_speech_split(paths, indices),
    }
    return indices, manifest


SPEECH_PREPROCESSING = "logmel64_fft1024_hop250_per_clip_db80_v1"


def cached_speech_split(payload) -> tuple[dict[str, np.ndarray], dict]:
    """Check saved row identities, official groups and the complete split audit."""
    import json

    required = {"filenames", "speaker_ids", "official_partition", "split_manifest", "labels"}
    required.update(f"{name}_indices" for name in PARTITIONS)
    if not required <= set(payload.keys()):
        raise ValueError(
            "Speaker-disjoint cache is missing split metadata; rebuild the separate cache"
        )
    paths, labels, speakers = _metadata(payload["filenames"].tolist())
    partition = payload["official_partition"]
    manifest = json.loads(str(payload["split_manifest"]))
    if partition.shape != (len(paths),) or not np.isin(partition, [0, 1, 2]).all():
        raise ValueError("Invalid official partition codes")
    if not np.array_equal(payload["speaker_ids"], speakers):
        raise ValueError("Saved speaker IDs do not match filenames")
    classes = sorted(set(labels))
    expected_labels = np.asarray([classes.index(word) for word in labels], dtype=np.int64)
    if not np.array_equal(payload["labels"], expected_labels):
        raise ValueError("Saved labels do not match keyword filenames")
    saved = {name: np.asarray(payload[f"{name}_indices"]) for name in PARTITIONS}
    audit_speech_split(paths, saved)  # Validate indices before using them.
    indices, expected = official_speech_split(
        paths,
        [paths[i] for i in np.flatnonzero(partition == 1)],
        [paths[i] for i in np.flatnonzero(partition == 2)],
        train_size=len(saved["train"]),
        training_files=[paths[i] for i in saved["train"]],
    )
    for name in PARTITIONS:
        if not np.array_equal(saved[name], indices[name]):
            raise ValueError(f"Saved {name} indices differ from official speaker split")
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Saved split audit does not match cache metadata")
    if manifest.get("preprocessing") != SPEECH_PREPROCESSING or "split_seed" not in manifest:
        raise ValueError("Unrecognized speech preprocessing or missing split seed")
    return indices, manifest
