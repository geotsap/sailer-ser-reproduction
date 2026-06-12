#!/usr/bin/env python3
"""
Create MSP-Podcast train/validation/test manifests from the raw annotation files.

Expected default raw layout:

    data/raw/msp_podcast/
    ├── Audio/
    ├── Labels.txt        # or labels.txt
    ├── Partitions.txt    # or Partition.txt / partitions.txt
    └── Speaker_ids.txt   # optional

Outputs:

    data/manifests/msp_podcast/
    ├── all.csv
    ├── train.csv
    ├── validation.csv
    ├── test.csv
    ├── unassigned.csv         # only if some labeled files are not in the partition file
    └── label_schema.json

The target columns are soft-label distributions computed from individual primary
emotion annotations, so samples with consensus code X / No Agreement can still be
used for distribution learning.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PRIMARY_LABELS: List[str] = [
    "Angry",
    "Sad",
    "Happy",
    "Surprise",
    "Fear",
    "Disgust",
    "Contempt",
    "Neutral",
    "Other",
]

SECONDARY_LABELS: List[str] = [
    "Angry",
    "Sad",
    "Happy",
    "Amused",
    "Neutral",
    "Frustrated",
    "Depressed",
    "Surprise",
    "Concerned",
    "Disgust",
    "Disappointed",
    "Excited",
    "Confused",
    "Annoyed",
    "Fear",
    "Contempt",
    "Other",
]

CONSENSUS_CODE_TO_LABEL: Dict[str, str] = {
    "A": "Angry",
    "S": "Sad",
    "H": "Happy",
    "U": "Surprise",
    "F": "Fear",
    "D": "Disgust",
    "C": "Contempt",
    "N": "Neutral",
    "O": "Other",
    "X": "No Agreement",
}

PRIMARY_ALIASES: Dict[str, str] = {
    "angry": "Angry",
    "anger": "Angry",
    "sad": "Sad",
    "sadness": "Sad",
    "happy": "Happy",
    "happiness": "Happy",
    "surprise": "Surprise",
    "surprised": "Surprise",
    "fear": "Fear",
    "fearful": "Fear",
    "disgust": "Disgust",
    "disgusted": "Disgust",
    "contempt": "Contempt",
    "neutral": "Neutral",
    "other": "Other",
}

SECONDARY_ALIASES: Dict[str, str] = {
    **PRIMARY_ALIASES,
    "amused": "Amused",
    "frustrated": "Frustrated",
    "depressed": "Depressed",
    "concerned": "Concerned",
    "disappointed": "Disappointed",
    "excited": "Excited",
    "confused": "Confused",
    "annoyed": "Annoyed",
}

SPLIT_ALIASES: Dict[str, str] = {
    "train": "train",
    "training": "train",
    "development": "validation",
    "develop": "validation",
    "dev": "validation",
    "validation": "validation",
    "valid": "validation",
    "val": "validation",
    "test": "test",
    "testing": "test",
}


class ManifestError(RuntimeError):
    """Raised for malformed inputs that should stop manifest creation."""


def split_semicolon(line: str) -> List[str]:
    """Split a semicolon-delimited annotation line and remove empty trailing cells."""
    return [part.strip() for part in line.strip().split(";") if part.strip()]


def strip_parenthetical(label: str) -> str:
    """Convert e.g. Other(confused) to Other before canonical mapping."""
    return re.sub(r"\s*\(.*?\)\s*", "", label.strip()).strip()


def normalize_primary_label(raw: str) -> str:
    base = strip_parenthetical(raw)
    key = base.strip().lower()
    return PRIMARY_ALIASES.get(key, "Other" if key.startswith("other") else base)


def normalize_secondary_label(raw: str) -> str:
    base = strip_parenthetical(raw)
    key = base.strip().lower()
    return SECONDARY_ALIASES.get(key, "Other" if key.startswith("other") else base)


def parse_dimension_fields(fields: Iterable[str]) -> Dict[str, Optional[float]]:
    dims: Dict[str, Optional[float]] = {"A": None, "V": None, "D": None}
    for field in fields:
        match = re.match(r"^\s*([AVD])\s*:\s*([-+]?\d+(?:\.\d+)?)\s*$", field)
        if match:
            dims[match.group(1)] = float(match.group(2))
    return dims


def mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def safe_float(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.6f}"


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def find_first_existing(raw_dir: Path, candidates: Iterable[str], required: bool = True) -> Optional[Path]:
    """Find a file in raw_dir by common names, allowing case-insensitive fallback."""
    for name in candidates:
        path = raw_dir / name
        if path.exists():
            return path

    lowered = {p.name.lower(): p for p in raw_dir.iterdir()} if raw_dir.exists() else {}
    for name in candidates:
        match = lowered.get(name.lower())
        if match is not None:
            return match

    if required:
        raise ManifestError(
            f"Could not find any of {list(candidates)} under raw directory: {raw_dir}"
        )
    return None


def parse_labels_file(labels_path: Path) -> Dict[str, Dict[str, Any]]:
    text = labels_path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\s*\n", text.strip())
    records: Dict[str, Dict[str, Any]] = {}

    for block_index, block in enumerate(blocks, start=1):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        header = split_semicolon(lines[0])
        if len(header) < 2:
            raise ManifestError(
                f"Malformed label header in block {block_index}: {lines[0]!r}"
            )

        filename = header[0]
        consensus_code = header[1].strip().upper()
        consensus_label = CONSENSUS_CODE_TO_LABEL.get(consensus_code, consensus_code)
        header_dims = parse_dimension_fields(header[2:])

        annotations: List[Dict[str, Any]] = []
        primary_counts: Counter[str] = Counter()
        secondary_counts: Counter[str] = Counter()
        worker_ids: List[str] = []
        worker_primary_raw: List[str] = []
        worker_primary_canonical: List[str] = []
        worker_secondary_raw: List[List[str]] = []
        worker_avd: Dict[str, List[float]] = {"A": [], "V": [], "D": []}

        for line_number, line in enumerate(lines[1:], start=2):
            fields = split_semicolon(line)
            if len(fields) < 3:
                print(
                    f"Warning: skipping malformed worker line in block {block_index}, "
                    f"line {line_number}: {line!r}",
                    file=sys.stderr,
                )
                continue

            worker_id = fields[0]
            primary_raw = fields[1]
            secondary_raw_field = fields[2]
            dims = parse_dimension_fields(fields[3:])

            primary = normalize_primary_label(primary_raw)
            if primary not in PRIMARY_LABELS:
                primary = "Other"

            secondary_raw_labels = [
                item.strip() for item in secondary_raw_field.split(",") if item.strip()
            ]
            secondary_labels: List[str] = []
            for raw_label in secondary_raw_labels:
                secondary = normalize_secondary_label(raw_label)
                if secondary not in SECONDARY_LABELS:
                    secondary = "Other"
                secondary_labels.append(secondary)

            worker_ids.append(worker_id)
            worker_primary_raw.append(primary_raw)
            worker_primary_canonical.append(primary)
            worker_secondary_raw.append(secondary_raw_labels)
            primary_counts[primary] += 1
            for secondary in secondary_labels:
                secondary_counts[secondary] += 1
            for key in ("A", "V", "D"):
                if dims[key] is not None:
                    worker_avd[key].append(float(dims[key]))

            annotations.append(
                {
                    "worker_id": worker_id,
                    "primary_raw": primary_raw,
                    "primary": primary,
                    "secondary_raw": secondary_raw_labels,
                    "secondary": secondary_labels,
                    "arousal": dims["A"],
                    "valence": dims["V"],
                    "dominance": dims["D"],
                }
            )

        num_annotations = len(annotations)
        if num_annotations > 0:
            primary_distribution = {
                label: primary_counts[label] / num_annotations for label in PRIMARY_LABELS
            }
            secondary_rates = {
                label: secondary_counts[label] / num_annotations for label in SECONDARY_LABELS
            }
        else:
            primary_distribution = {label: 0.0 for label in PRIMARY_LABELS}
            secondary_rates = {label: 0.0 for label in SECONDARY_LABELS}

        records[filename] = {
            "filename": filename,
            "consensus_code": consensus_code,
            "consensus_label": consensus_label,
            "has_consensus": consensus_code != "X",
            "consensus_arousal": header_dims["A"],
            "consensus_valence": header_dims["V"],
            "consensus_dominance": header_dims["D"],
            "mean_worker_arousal": mean(worker_avd["A"]),
            "mean_worker_valence": mean(worker_avd["V"]),
            "mean_worker_dominance": mean(worker_avd["D"]),
            "num_annotations": num_annotations,
            "primary_counts": {label: int(primary_counts[label]) for label in PRIMARY_LABELS},
            "primary_distribution": primary_distribution,
            "secondary_counts": {label: int(secondary_counts[label]) for label in SECONDARY_LABELS},
            "secondary_rates": secondary_rates,
            "worker_ids": worker_ids,
            "worker_primary_raw": worker_primary_raw,
            "worker_primary": worker_primary_canonical,
            "worker_secondary_raw": worker_secondary_raw,
            "annotations": annotations,
        }

    return records


def parse_partitions_file(partitions_path: Path) -> Dict[str, str]:
    partition_map: Dict[str, str] = {}

    with partitions_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = split_semicolon(line)
            if len(fields) < 2:
                print(
                    f"Warning: skipping malformed partition line {line_number}: {raw_line!r}",
                    file=sys.stderr,
                )
                continue

            split_raw, filename = fields[0], fields[1]
            split = SPLIT_ALIASES.get(split_raw.strip().lower())
            if split is None:
                print(
                    f"Warning: unknown split {split_raw!r} on line {line_number}; "
                    "using 'unassigned'.",
                    file=sys.stderr,
                )
                split = "unassigned"
            partition_map[filename] = split

    return partition_map


def parse_speaker_file(speaker_path: Optional[Path]) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
    """Return filename->speaker_id and speaker_id->metadata maps.

    Speaker_ids.txt has two sections: speaker gender rows, then filename to speaker
    number rows after a line of asterisks.
    """
    if speaker_path is None or not speaker_path.exists():
        return {}, {}

    filename_to_speaker: Dict[str, str] = {}
    speaker_meta: Dict[str, Dict[str, str]] = {}
    in_assignment_section = False

    with speaker_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if set(line) == {"*"}:
                in_assignment_section = True
                continue

            fields = split_semicolon(line)
            if len(fields) < 2:
                continue

            if not in_assignment_section:
                speaker_id = fields[0].strip()
                raw_meta = fields[1].strip()
                raw_lower = raw_meta.lower()
                if "female" in raw_lower:
                    gender = "Female"
                elif "male" in raw_lower:
                    gender = "Male"
                else:
                    gender = raw_meta
                speaker_meta[speaker_id] = {"gender": gender, "raw": raw_meta}
            else:
                filename = fields[0].strip()
                speaker_num = fields[1].strip()
                speaker_id = (
                    speaker_num if speaker_num.lower().startswith("speaker_") else f"Speaker_{speaker_num}"
                )
                filename_to_speaker[filename] = speaker_id

    return filename_to_speaker, speaker_meta


def build_manifest_rows(
    label_records: Dict[str, Dict[str, Any]],
    partition_map: Dict[str, str],
    filename_to_speaker: Dict[str, str],
    speaker_meta: Dict[str, Dict[str, str]],
    audio_dir: Path,
    strict_audio: bool = False,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    for filename in sorted(label_records.keys()):
        record = label_records[filename]
        split = partition_map.get(filename, "unassigned")
        audio_path = audio_dir / filename
        audio_exists = audio_path.exists()
        if strict_audio and not audio_exists:
            raise ManifestError(f"Audio file listed in labels but not found: {audio_path}")

        speaker_id = filename_to_speaker.get(filename, "")
        gender = speaker_meta.get(speaker_id, {}).get("gender", "") if speaker_id else ""

        row: Dict[str, str] = {
            "utt_id": Path(filename).stem,
            "filename": filename,
            "audio_path": str(audio_path),
            "audio_exists": str(audio_exists),
            "split": split,
            "speaker_id": speaker_id,
            "speaker_gender": gender,
            "consensus_code": record["consensus_code"],
            "consensus_label": record["consensus_label"],
            "has_consensus": str(bool(record["has_consensus"])),
            "num_annotations": str(record["num_annotations"]),
            "consensus_arousal": safe_float(record["consensus_arousal"]),
            "consensus_valence": safe_float(record["consensus_valence"]),
            "consensus_dominance": safe_float(record["consensus_dominance"]),
            "mean_worker_arousal": safe_float(record["mean_worker_arousal"]),
            "mean_worker_valence": safe_float(record["mean_worker_valence"]),
            "mean_worker_dominance": safe_float(record["mean_worker_dominance"]),
            "primary_counts_json": json_compact(record["primary_counts"]),
            "primary_distribution_json": json_compact(record["primary_distribution"]),
            "secondary_counts_json": json_compact(record["secondary_counts"]),
            "secondary_rates_json": json_compact(record["secondary_rates"]),
            "worker_ids_json": json_compact(record["worker_ids"]),
            "worker_primary_json": json_compact(record["worker_primary"]),
            "worker_primary_raw_json": json_compact(record["worker_primary_raw"]),
            "worker_secondary_raw_json": json_compact(record["worker_secondary_raw"]),
        }

        for label in PRIMARY_LABELS:
            key = label.lower()
            row[f"primary_count_{key}"] = str(record["primary_counts"][label])
            row[f"target_{key}"] = f"{record['primary_distribution'][label]:.8f}"

        for label in SECONDARY_LABELS:
            key = label.lower()
            row[f"secondary_count_{key}"] = str(record["secondary_counts"][label])
            row[f"secondary_rate_{key}"] = f"{record['secondary_rates'][label]:.8f}"

        rows.append(row)

    return rows


def get_fieldnames() -> List[str]:
    base_fields = [
        "utt_id",
        "filename",
        "audio_path",
        "audio_exists",
        "split",
        "speaker_id",
        "speaker_gender",
        "consensus_code",
        "consensus_label",
        "has_consensus",
        "num_annotations",
        "consensus_arousal",
        "consensus_valence",
        "consensus_dominance",
        "mean_worker_arousal",
        "mean_worker_valence",
        "mean_worker_dominance",
    ]
    primary_fields: List[str] = []
    for label in PRIMARY_LABELS:
        key = label.lower()
        primary_fields.extend([f"primary_count_{key}", f"target_{key}"])

    secondary_fields: List[str] = []
    for label in SECONDARY_LABELS:
        key = label.lower()
        secondary_fields.extend([f"secondary_count_{key}", f"secondary_rate_{key}"])

    json_fields = [
        "primary_counts_json",
        "primary_distribution_json",
        "secondary_counts_json",
        "secondary_rates_json",
        "worker_ids_json",
        "worker_primary_json",
        "worker_primary_raw_json",
        "worker_secondary_raw_json",
    ]
    return base_fields + primary_fields + secondary_fields + json_fields


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_schema(path: Path, labels_path: Path, partitions_path: Path, speaker_path: Optional[Path]) -> None:
    schema = {
        "primary_labels": PRIMARY_LABELS,
        "secondary_labels": SECONDARY_LABELS,
        "consensus_code_to_label": CONSENSUS_CODE_TO_LABEL,
        "target_columns": [f"target_{label.lower()}" for label in PRIMARY_LABELS],
        "secondary_rate_columns": [f"secondary_rate_{label.lower()}" for label in SECONDARY_LABELS],
        "source_files": {
            "labels": str(labels_path),
            "partitions": str(partitions_path),
            "speaker_ids": str(speaker_path) if speaker_path is not None else None,
        },
        "notes": [
            "Primary target_* columns are normalized individual primary annotation counts.",
            "Secondary_rate_* columns are multi-label rates: count(label) / num_annotations.",
            "Consensus code X is retained as No Agreement; its primary target distribution is still computed from workers.",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def summarize(rows: List[Dict[str, str]]) -> str:
    split_counts = Counter(row["split"] for row in rows)
    consensus_counts = Counter(row["consensus_label"] for row in rows)
    lines = ["Manifest creation summary:"]
    lines.append(f"  total labeled rows: {len(rows)}")
    lines.append("  split counts:")
    for split in ["train", "validation", "test", "unassigned"]:
        if split_counts[split]:
            lines.append(f"    {split}: {split_counts[split]}")
    lines.append("  consensus label counts:")
    for label, count in consensus_counts.most_common():
        lines.append(f"    {label}: {count}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create MSP-Podcast train/validation/test manifests from Labels.txt and Partitions.txt."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/msp_podcast"),
        help="Directory containing Labels.txt, Partitions.txt/Partition.txt, Speaker_ids.txt, and Audio/.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/manifests/msp_podcast"),
        help="Directory where manifest CSV files will be written.",
    )
    parser.add_argument(
        "--labels-file",
        type=Path,
        default=None,
        help="Explicit path to Labels.txt. If omitted, common names are searched in --raw-dir.",
    )
    parser.add_argument(
        "--partitions-file",
        type=Path,
        default=None,
        help="Explicit path to Partitions.txt/Partition.txt. If omitted, common names are searched in --raw-dir.",
    )
    parser.add_argument(
        "--speaker-file",
        type=Path,
        default=None,
        help="Explicit path to Speaker_ids.txt. If omitted, common names are searched in --raw-dir; optional.",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="Explicit audio directory. Defaults to --raw-dir/Audio.",
    )
    parser.add_argument(
        "--strict-audio",
        action="store_true",
        help="Fail if an audio file listed in Labels.txt is not present under the audio directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir: Path = args.raw_dir
    out_dir: Path = args.out_dir
    audio_dir: Path = args.audio_dir if args.audio_dir is not None else raw_dir / "Audio"

    try:
        labels_path = args.labels_file or find_first_existing(
            raw_dir, ["Labels.txt", "labels.txt", "LABELS.txt"]
        )
        partitions_path = args.partitions_file or find_first_existing(
            raw_dir,
            ["Partitions.txt", "Partition.txt", "partitions.txt", "partition.txt"],
        )
        speaker_path = args.speaker_file or find_first_existing(
            raw_dir,
            ["Speaker_ids.txt", "speaker_ids.txt", "Speaker_Ids.txt"],
            required=False,
        )

        assert labels_path is not None
        assert partitions_path is not None

        label_records = parse_labels_file(labels_path)
        partition_map = parse_partitions_file(partitions_path)
        filename_to_speaker, speaker_meta = parse_speaker_file(speaker_path)
        rows = build_manifest_rows(
            label_records=label_records,
            partition_map=partition_map,
            filename_to_speaker=filename_to_speaker,
            speaker_meta=speaker_meta,
            audio_dir=audio_dir,
            strict_audio=args.strict_audio,
        )

        fieldnames = get_fieldnames()
        write_csv(out_dir / "all.csv", rows, fieldnames)
        for split in ["train", "validation", "test", "unassigned"]:
            split_rows = [row for row in rows if row["split"] == split]
            if split_rows or split != "unassigned":
                write_csv(out_dir / f"{split}.csv", split_rows, fieldnames)

        partition_filenames = set(partition_map.keys())
        label_filenames = set(label_records.keys())
        partition_without_labels = sorted(partition_filenames - label_filenames)
        if partition_without_labels:
            (out_dir / "partition_without_labels.txt").write_text(
                "\n".join(partition_without_labels) + "\n", encoding="utf-8"
            )
            print(
                f"Warning: {len(partition_without_labels)} partition entries had no label block. "
                f"Wrote {out_dir / 'partition_without_labels.txt'}",
                file=sys.stderr,
            )

        write_schema(out_dir / "label_schema.json", labels_path, partitions_path, speaker_path)
        print(summarize(rows))
        print(f"\nWrote manifests to: {out_dir}")
        return 0

    except ManifestError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
