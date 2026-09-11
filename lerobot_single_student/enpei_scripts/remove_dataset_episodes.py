#!/usr/bin/env python

"""Create a clean LeRobot v2.1 dataset while excluding selected episodes.

The source dataset is never modified. Kept videos are copied without
re-encoding, while episode and global frame indices are made contiguous in the
new dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def read_jsonlines(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonlines(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def sequence_stats(start: int, length: int) -> dict[str, list[int | float]]:
    return {
        "min": [start],
        "max": [start + length - 1],
        "mean": [start + (length - 1) / 2],
        "std": [math.sqrt((length**2 - 1) / 12)],
        "count": [length],
    }


def constant_stats(value: int, length: int) -> dict[str, list[int | float]]:
    return {
        "min": [value],
        "max": [value],
        "mean": [float(value)],
        "std": [0.0],
        "count": [length],
    }


def replace_int_column(table: pa.Table, name: str, values: list[int]) -> pa.Table:
    column_index = table.schema.get_field_index(name)
    if column_index < 0:
        raise ValueError(f"Required parquet column is missing: {name}")
    column_type = table.schema.field(column_index).type
    return table.set_column(column_index, name, pa.array(values, type=column_type))


def clean_dataset(source_root: Path, output_root: Path, removed: set[int]) -> None:
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()

    if source_root == output_root:
        raise ValueError("The output directory must be different from the source directory.")
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source dataset does not exist: {source_root}")
    if output_root.exists():
        raise FileExistsError(f"Output directory already exists: {output_root}")

    meta_dir = source_root / "meta"
    info = json.loads((meta_dir / "info.json").read_text(encoding="utf-8"))
    episodes = read_jsonlines(meta_dir / "episodes.jsonl")
    episode_stats = read_jsonlines(meta_dir / "episodes_stats.jsonl")

    episode_by_index = {record["episode_index"]: record for record in episodes}
    stats_by_index = {record["episode_index"]: record for record in episode_stats}
    available = set(episode_by_index)
    missing = removed - available
    if missing:
        raise ValueError(f"Episodes are not present in the source dataset: {sorted(missing)}")

    chunks_size = info["chunks_size"]
    data_pattern = info["data_path"]
    video_pattern = info["video_path"]
    video_keys = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    kept = sorted(available - removed)

    temporary_root = output_root.parent / f".{output_root.name}.tmp-{os.getpid()}"
    if temporary_root.exists():
        raise FileExistsError(f"Temporary output directory already exists: {temporary_root}")
    temporary_root.mkdir(parents=True)

    new_episodes: list[dict] = []
    new_episode_stats: list[dict] = []
    global_frame_index = 0

    for new_episode_index, old_episode_index in enumerate(kept):
        episode = dict(episode_by_index[old_episode_index])
        episode_length = episode["length"]

        old_data_path = source_root / data_pattern.format(
            episode_chunk=old_episode_index // chunks_size,
            episode_index=old_episode_index,
        )
        new_data_path = temporary_root / data_pattern.format(
            episode_chunk=new_episode_index // chunks_size,
            episode_index=new_episode_index,
        )
        if not old_data_path.is_file():
            raise FileNotFoundError(f"Missing parquet file: {old_data_path}")

        table = pq.read_table(old_data_path)
        if table.num_rows != episode_length:
            raise ValueError(
                f"Episode {old_episode_index} length mismatch: "
                f"metadata={episode_length}, parquet={table.num_rows}"
            )
        table = replace_int_column(table, "episode_index", [new_episode_index] * episode_length)
        table = replace_int_column(
            table,
            "index",
            list(range(global_frame_index, global_frame_index + episode_length)),
        )
        new_data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, new_data_path)

        for video_key in video_keys:
            old_video_path = source_root / video_pattern.format(
                episode_chunk=old_episode_index // chunks_size,
                episode_index=old_episode_index,
                video_key=video_key,
            )
            new_video_path = temporary_root / video_pattern.format(
                episode_chunk=new_episode_index // chunks_size,
                episode_index=new_episode_index,
                video_key=video_key,
            )
            if not old_video_path.is_file():
                raise FileNotFoundError(f"Missing video file: {old_video_path}")
            new_video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_video_path, new_video_path)

        episode["episode_index"] = new_episode_index
        new_episodes.append(episode)

        stats_record = json.loads(json.dumps(stats_by_index[old_episode_index]))
        stats_record["episode_index"] = new_episode_index
        if "episode_index" in stats_record["stats"]:
            stats_record["stats"]["episode_index"] = constant_stats(new_episode_index, episode_length)
        if "index" in stats_record["stats"]:
            stats_record["stats"]["index"] = sequence_stats(global_frame_index, episode_length)
        new_episode_stats.append(stats_record)

        global_frame_index += episode_length
        print(
            f"Kept episode {old_episode_index} as {new_episode_index} "
            f"({episode_length} frames)"
        )

    info["total_episodes"] = len(kept)
    info["total_frames"] = global_frame_index
    info["total_videos"] = len(kept) * len(video_keys)
    info["total_chunks"] = math.ceil(len(kept) / chunks_size) if kept else 0
    info["splits"] = {"train": f"0:{len(kept)}"}

    output_meta = temporary_root / "meta"
    output_meta.mkdir(parents=True, exist_ok=True)
    (output_meta / "info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )
    write_jsonlines(output_meta / "episodes.jsonl", new_episodes)
    write_jsonlines(output_meta / "episodes_stats.jsonl", new_episode_stats)
    shutil.copy2(meta_dir / "tasks.jsonl", output_meta / "tasks.jsonl")

    temporary_root.rename(output_root)
    print(f"Removed episodes: {sorted(removed)}")
    print(f"Clean dataset: {output_root}")
    print(f"Episodes: {len(kept)}, frames: {global_frame_index}, videos: {info['total_videos']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--remove-episodes", type=int, nargs="+", required=True)
    args = parser.parse_args()

    clean_dataset(args.source_root, args.output_root, set(args.remove_episodes))


if __name__ == "__main__":
    main()
