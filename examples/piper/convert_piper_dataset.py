from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import jsonlines
import pyarrow.parquet as pq

DEFAULT_CHUNK_SIZE = 1000
DATA_PATH_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH_TEMPLATE = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
MARKER_PATH = ".piper_source.json"
PIPER_ACTION_KEY = "action"
PIPER_STATE_KEY = "observation.state"
PIPER_BASE_IMAGE_KEY = "observation.images.base"
PIPER_WRIST_IMAGE_KEY = "observation.images.wrist"
REQUIRED_VIDEO_KEYS = (PIPER_BASE_IMAGE_KEY, PIPER_WRIST_IMAGE_KEY)
MIN_VIDEO_DURATION = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage a Piper LeRobot dataset for openpi. Supports validating v2.1 inputs or converting v3.0."
    )
    parser.add_argument("--src", type=Path, required=True, help="Source LeRobot dataset root.")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Canonical output repo id. Defaults to '<parent>/<name>' inferred from --src.",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        default=None,
        help="Destination base directory. Defaults to the base directory of --src.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite any existing staged dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = args.src.expanduser().resolve()
    repo_id = args.repo_id or infer_repo_id(src)
    dst_root = args.dst_root.expanduser().resolve() if args.dst_root is not None else infer_dst_root(src)
    dst = dst_root / repo_id

    if not src.exists():
        raise FileNotFoundError(f"Source dataset not found: {src}")

    source_info = load_json(src / "meta" / "info.json")
    source_version = source_info.get("codebase_version")
    tasks = load_tasks(src, source_version)
    validate_piper_metadata(source_info, tasks, source_version)

    if source_version == "v2.1" and src == dst:
        print(f"Validated existing staged dataset in place: {dst}")
        return

    ensure_destination(dst, force=args.force)

    if source_version == "v2.1":
        stage_v21_dataset(src, dst, repo_id, source_version)
        print(f"Staged Piper dataset at {dst}")
        return

    if source_version != "v3.0":
        raise ValueError(f"Unsupported dataset version {source_version!r}. Expected 'v2.1' or 'v3.0'.")

    convert_v30_to_v21(src, dst, repo_id, source_info, tasks)
    print(f"Converted Piper dataset to openpi-compatible v2.1 format at {dst}")


def infer_repo_id(src: Path) -> str:
    if src.parent.name:
        return f"{src.parent.name}/{src.name}"
    return src.name


def infer_dst_root(src: Path) -> Path:
    if src.parent.name:
        return src.parent.parent
    return src.parent


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required metadata file not found: {path}")
    return json.loads(path.read_text())


def load_tasks(src: Path, source_version: str | None) -> list[dict]:
    if source_version == "v3.0":
        tasks_path = src / "meta" / "tasks.parquet"
        if not tasks_path.exists():
            raise FileNotFoundError(f"Required task metadata file not found: {tasks_path}")
        tasks = pq.read_table(tasks_path).to_pylist()
    else:
        tasks_path = src / "meta" / "tasks.jsonl"
        if not tasks_path.exists():
            raise FileNotFoundError(f"Required task metadata file not found: {tasks_path}")
        with jsonlines.open(tasks_path) as reader:
            tasks = list(reader)
    if not tasks:
        raise ValueError(f"No tasks found in {tasks_path}")
    return tasks


def validate_piper_metadata(info: dict, tasks: list[dict], source_version: str | None) -> None:
    features = info.get("features", {})
    action_feature = features.get(PIPER_ACTION_KEY)
    state_feature = features.get(PIPER_STATE_KEY)
    if action_feature is None:
        raise ValueError(f"Dataset is missing required feature {PIPER_ACTION_KEY!r}.")
    if state_feature is None:
        raise ValueError(f"Dataset is missing required feature {PIPER_STATE_KEY!r}.")

    if tuple(action_feature.get("shape", ())) != (7,):
        raise ValueError(f"{PIPER_ACTION_KEY} must have shape [7], found {action_feature.get('shape')}.")
    if tuple(state_feature.get("shape", ())) != (7,):
        raise ValueError(f"{PIPER_STATE_KEY} must have shape [7], found {state_feature.get('shape')}.")

    for key in REQUIRED_VIDEO_KEYS:
        feature = features.get(key)
        if feature is None:
            raise ValueError(f"Dataset is missing required camera feature {key!r}.")
        if feature.get("dtype") not in {"image", "video"}:
            raise ValueError(f"{key} must be stored as image or video, found {feature.get('dtype')!r}.")

    if not all(task.get("task") for task in tasks):
        raise ValueError(f"All tasks in {source_version or 'dataset'} metadata must contain a non-empty 'task' field.")


def _to_serializable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value


def ensure_destination(dst: Path, *, force: bool) -> None:
    marker_path = dst / MARKER_PATH
    if dst.exists():
        if not force and marker_path.exists():
            marker = load_json(marker_path)
            print(
                f"Using existing staged dataset at {dst} "
                f"(source={marker.get('source')}, version={marker.get('codebase_version')})."
            )
            raise SystemExit(0)
        if not force:
            raise FileExistsError(f"Destination already exists: {dst}. Re-run with --force to overwrite.")
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)


def stage_v21_dataset(src: Path, dst: Path, repo_id: str, source_version: str) -> None:
    validate_v21_layout(src)
    work_root = Path(tempfile.mkdtemp(prefix=f".{dst.name}.tmp.", dir=dst.parent))
    staging_dir = work_root / "dataset"
    try:
        shutil.copytree(src, staging_dir)
        write_marker(staging_dir, src, repo_id, source_version)
        shutil.move(staging_dir, dst)
    finally:
        shutil.rmtree(work_root, ignore_errors=True)


def validate_v21_layout(src: Path) -> None:
    required_paths = [
        src / "meta" / "episodes.jsonl",
        src / "meta" / "episodes_stats.jsonl",
        src / "meta" / "stats.json",
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required v2.1 metadata file not found: {path}")

    if first_path(src.glob("data/chunk-*/*.parquet")) is None:
        raise FileNotFoundError(f"No episode parquet files found under {src / 'data'}")

    for video_key in REQUIRED_VIDEO_KEYS:
        if first_path((src / "videos").glob(f"chunk-*/{video_key}/episode_*.mp4")) is None:
            raise FileNotFoundError(f"No staged videos found for {video_key!r} under {src / 'videos'}")


def load_episode_records(src: Path) -> list[dict[str, Any]]:
    episodes_dir = src / "meta" / "episodes"
    pq_paths = sorted(episodes_dir.glob("chunk-*/file-*.parquet"))
    if not pq_paths:
        raise FileNotFoundError(f"No episode parquet files found in {episodes_dir}.")

    records: list[dict[str, Any]] = []
    for pq_path in pq_paths:
        records.extend(pq.read_table(pq_path).to_pylist())

    records.sort(key=lambda rec: int(rec["episode_index"]))
    if not records:
        raise ValueError(f"No episodes found in {episodes_dir}.")
    return records


def convert_info(source_info: dict, new_root: Path, episode_records: list[dict[str, Any]], video_keys: list[str]) -> None:
    info = json.loads(json.dumps(source_info))
    total_episodes = int(info.get("total_episodes") or len(episode_records))
    chunks_size = int(info.get("chunks_size", DEFAULT_CHUNK_SIZE))

    info["codebase_version"] = "v2.1"
    info["data_path"] = DATA_PATH_TEMPLATE
    info["video_path"] = VIDEO_PATH_TEMPLATE if video_keys else None
    info.pop("data_files_size_in_mb", None)
    info.pop("video_files_size_in_mb", None)
    info["total_chunks"] = math.ceil(total_episodes / chunks_size) if total_episodes > 0 else 0
    info["total_videos"] = total_episodes * len(video_keys)

    write_json(info, new_root / "meta" / "info.json")


def copy_global_stats(src: Path, new_root: Path) -> None:
    source_stats = src / "meta" / "stats.json"
    if source_stats.exists():
        target_stats = new_root / "meta" / "stats.json"
        target_stats.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_stats, target_stats)


def convert_tasks(tasks: list[dict], new_root: Path) -> None:
    out_path = new_root / "meta" / "tasks.jsonl"
    rows = [
        {
            "task_index": int(task["task_index"]),
            "task": str(task["task"]).strip(),
        }
        for task in sorted(tasks, key=lambda row: int(row["task_index"]))
    ]
    write_jsonl(rows, out_path)


def _group_episodes_by_data_file(
    episode_records: list[dict[str, Any]],
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in episode_records:
        grouped[(int(record["data/chunk_index"]), int(record["data/file_index"]))].append(record)
    return grouped


def convert_data(
    src: Path,
    new_root: Path,
    episode_records: list[dict[str, Any]],
    source_data_path: str,
    chunks_size: int,
) -> None:
    grouped = _group_episodes_by_data_file(episode_records)
    print(f"Converting {len(grouped)} consolidated data parquet file(s)...")

    for (chunk_idx, file_idx), records in grouped.items():
        source_path = src / source_data_path.format(chunk_index=chunk_idx, file_index=file_idx)
        if not source_path.exists():
            raise FileNotFoundError(f"Expected source parquet file not found: {source_path}")

        table = pq.read_table(source_path)
        records = sorted(records, key=lambda rec: int(rec["dataset_from_index"]))
        file_offset = int(records[0]["dataset_from_index"])

        for record in records:
            episode_index = int(record["episode_index"])
            start = int(record["dataset_from_index"]) - file_offset
            stop = int(record["dataset_to_index"]) - file_offset
            length = stop - start
            if length <= 0:
                raise ValueError(
                    "Invalid episode length computed during data conversion: "
                    f"episode_index={episode_index}, length={length}"
                )

            episode_table = table.slice(start, length)
            dest_chunk = episode_index // chunks_size
            dest_path = new_root / DATA_PATH_TEMPLATE.format(
                episode_chunk=dest_chunk,
                episode_index=episode_index,
            )
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(episode_table, dest_path)


def _group_episodes_by_video_file(
    episode_records: list[dict[str, Any]], video_key: str
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    chunk_column = f"videos/{video_key}/chunk_index"
    file_column = f"videos/{video_key}/file_index"

    for record in episode_records:
        chunk_idx = record.get(chunk_column)
        file_idx = record.get(file_column)
        if chunk_idx is None or file_idx is None:
            continue
        grouped[(int(chunk_idx), int(file_idx))].append(record)

    return grouped


def _extract_video_segment(src: Path, dst: Path, start: float, end: float) -> None:
    duration = max(end - start, MIN_VIDEO_DURATION)
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.6f}",
        "-i",
        str(src),
        "-t",
        f"{duration:.6f}",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "1",
        "-y",
        str(dst),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=300, capture_output=True, text=True)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffmpeg timed out while splitting video '{src}' -> '{dst}'") from exc
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg executable not found; it is required for Piper dataset conversion.") from exc
    except subprocess.CalledProcessError as exc:
        error_msg = f"ffmpeg failed while splitting video '{src}' into '{dst}'"
        if exc.stderr:
            error_msg += f". Error: {exc.stderr.strip()}"
        raise RuntimeError(error_msg) from exc


def convert_videos(
    src: Path,
    new_root: Path,
    episode_records: list[dict[str, Any]],
    source_video_path: str | None,
    video_keys: list[str],
    chunks_size: int,
) -> None:
    if not video_keys:
        print("No video features detected; skipping video conversion.")
        return
    if source_video_path is None:
        raise ValueError("Source dataset is missing a video_path template in info.json.")

    total_segments = 0
    for video_key in video_keys:
        grouped = _group_episodes_by_video_file(episode_records, video_key)
        total_segments += sum(len(records) for records in grouped.values())
    print(f"Splitting {total_segments} video segment(s) across {len(video_keys)} camera stream(s)...")

    for video_key in video_keys:
        grouped = _group_episodes_by_video_file(episode_records, video_key)
        for (chunk_idx, file_idx), records in grouped.items():
            src_path = src / source_video_path.format(video_key=video_key, chunk_index=chunk_idx, file_index=file_idx)
            if not src_path.exists():
                raise FileNotFoundError(f"Expected source video file not found: {src_path}")

            records = sorted(records, key=lambda rec: float(rec[f"videos/{video_key}/from_timestamp"]))
            for record in records:
                episode_index = int(record["episode_index"])
                start = float(record[f"videos/{video_key}/from_timestamp"])
                end = float(record[f"videos/{video_key}/to_timestamp"])

                dest_chunk = episode_index // chunks_size
                dest_path = new_root / VIDEO_PATH_TEMPLATE.format(
                    episode_chunk=dest_chunk,
                    video_key=video_key,
                    episode_index=episode_index,
                )
                _extract_video_segment(src_path, dest_path, start=start, end=end)


def _extract_episode_stats(record: dict[str, Any]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for key, value in record.items():
        if not key.startswith("stats/"):
            continue
        parts = key.split("/")[1:]
        node = stats
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _to_serializable(value)
    return stats


def convert_episodes_metadata(new_root: Path, episode_records: list[dict[str, Any]]) -> None:
    episodes_path = new_root / "meta" / "episodes.jsonl"
    stats_path = new_root / "meta" / "episodes_stats.jsonl"

    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        jsonlines.open(episodes_path, mode="w") as episodes_writer,
        jsonlines.open(stats_path, mode="w") as stats_writer,
    ):
        for record in episode_records:
            legacy_episode = {
                key: _to_serializable(value)
                for key, value in record.items()
                if not key.startswith("data/")
                and not key.startswith("videos/")
                and not key.startswith("stats/")
                and not key.startswith("meta/")
                and key not in {"dataset_from_index", "dataset_to_index"}
            }
            if "length" not in legacy_episode:
                legacy_episode["length"] = int(record["dataset_to_index"]) - int(record["dataset_from_index"])

            episodes_writer.write(legacy_episode)
            stats_writer.write(
                {
                    "episode_index": int(record["episode_index"]),
                    "stats": _extract_episode_stats(record),
                }
            )


def convert_v30_to_v21(src: Path, dst: Path, repo_id: str, source_info: dict, tasks: list[dict]) -> None:
    video_keys = [
        key
        for key in REQUIRED_VIDEO_KEYS
        if source_info.get("features", {}).get(key, {}).get("dtype") == "video"
    ]
    if len(video_keys) != len(REQUIRED_VIDEO_KEYS):
        raise NotImplementedError(
            "Fast Piper v3->v2.1 conversion currently expects video-backed camera features for "
            f"{REQUIRED_VIDEO_KEYS}."
        )

    episode_records = load_episode_records(src)
    chunks_size = int(source_info.get("chunks_size", DEFAULT_CHUNK_SIZE))
    print(f"Loaded {len(episode_records)} episode metadata record(s) from {src}.")

    work_root = Path(tempfile.mkdtemp(prefix=f".{dst.name}.tmp.", dir=dst.parent))
    staging_dir = work_root / "dataset"
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("Converting metadata...")
        convert_info(source_info, staging_dir, episode_records, video_keys)
        copy_global_stats(src, staging_dir)
        convert_tasks(tasks, staging_dir)
        convert_data(src, staging_dir, episode_records, source_info["data_path"], chunks_size)
        convert_videos(src, staging_dir, episode_records, source_info.get("video_path"), video_keys, chunks_size)
        convert_episodes_metadata(staging_dir, episode_records)
        write_marker(staging_dir, src, repo_id, "v3.0")
        shutil.move(staging_dir, dst)
    finally:
        shutil.rmtree(work_root, ignore_errors=True)


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4))


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, mode="w") as writer:
        writer.write_all(rows)


def write_marker(dst: Path, src: Path, repo_id: str, codebase_version: str) -> None:
    write_json(
        {
            "source": str(src),
            "repo_id": repo_id,
            "codebase_version": codebase_version,
        },
        dst / MARKER_PATH,
    )


def first_path(paths) -> Path | None:
    return next(iter(paths), None)


if __name__ == "__main__":
    main()
