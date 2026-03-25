from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

from huggingface_hub.constants import HF_HOME
import jsonlines
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}
DATA_PATH_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH_TEMPLATE = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
MARKER_PATH = ".piper_source.json"
PIPER_ACTION_KEY = "action"
PIPER_STATE_KEY = "observation.state"
PIPER_BASE_IMAGE_KEY = "observation.images.base"
PIPER_WRIST_IMAGE_KEY = "observation.images.wrist"
REQUIRED_VIDEO_KEYS = (PIPER_BASE_IMAGE_KEY, PIPER_WRIST_IMAGE_KEY)
HF_LEROBOT_HOME = Path(os.getenv("HF_LEROBOT_HOME", Path(HF_HOME) / "lerobot")).expanduser()


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
        help="Destination HF_LEROBOT_HOME-like root. Defaults to $HF_LEROBOT_HOME.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite any existing staged dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = args.src.expanduser().resolve()
    repo_id = args.repo_id or infer_repo_id(src)
    dst_root = args.dst_root.expanduser().resolve() if args.dst_root is not None else HF_LEROBOT_HOME
    dst = dst_root / repo_id

    if not src.exists():
        raise FileNotFoundError(f"Source dataset not found: {src}")

    source_info = load_json(src / "meta" / "info.json")
    source_version = source_info.get("codebase_version")
    tasks = load_tasks(src)
    validate_piper_metadata(source_info, tasks)

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

    convert_v30_to_v21(src, dst, repo_id, source_info)
    print(f"Converted Piper dataset to openpi-compatible v2.1 format at {dst}")


def infer_repo_id(src: Path) -> str:
    if src.parent.name:
        return f"{src.parent.name}/{src.name}"
    return src.name


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required metadata file not found: {path}")
    return json.loads(path.read_text())


def load_tasks(src: Path) -> list[dict]:
    tasks_path = src / "meta" / "tasks.jsonl"
    if not tasks_path.exists():
        raise FileNotFoundError(f"Required task metadata file not found: {tasks_path}")
    with jsonlines.open(tasks_path) as reader:
        tasks = list(reader)
    if not tasks:
        raise ValueError(f"No tasks found in {tasks_path}")
    return tasks


def validate_piper_metadata(info: dict, tasks: list[dict]) -> None:
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
        raise ValueError("All tasks in tasks.jsonl must contain a non-empty 'task' field.")


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


def convert_v30_to_v21(src: Path, dst: Path, repo_id: str, source_info: dict) -> None:
    dataset_cls, metadata_cls, encode_video_frames = load_v3_lerobot()
    meta = metadata_cls(repo_id=repo_id, root=src)
    if meta.total_episodes <= 0:
        raise ValueError(f"No episodes found in source dataset: {src}")

    work_root = Path(tempfile.mkdtemp(prefix=f".{dst.name}.tmp.", dir=dst.parent))
    staging_dir = work_root / "dataset"
    staging_dir.mkdir(parents=True, exist_ok=True)

    tasks_to_index: dict[str, int] = {}
    episodes_rows: list[dict] = []
    episodes_stats_rows: list[dict] = []
    all_episode_stats: list[dict[str, dict[str, np.ndarray]]] = []
    output_features: dict | None = None
    total_frames = 0

    try:
        for episode_index in range(meta.total_episodes):
            dataset = dataset_cls(repo_id=repo_id, root=src, episodes=[episode_index], download_videos=True)
            episode_length = len(dataset)
            if episode_length <= 0:
                raise ValueError(f"Episode {episode_index} is empty.")

            episode_chunk = episode_index // DEFAULT_CHUNK_SIZE
            episode_task: str | None = None
            task_index: int | None = None

            actions: list[np.ndarray] = []
            states: list[np.ndarray] = []
            timestamps: list[float] = []
            frame_indices: list[int] = []
            task_indices: list[int] = []

            with tempfile.TemporaryDirectory(prefix=f"piper_ep_{episode_index:06d}_") as temp_dir_name:
                temp_dir = Path(temp_dir_name)
                frame_dirs = {
                    key: temp_dir / key
                    for key in REQUIRED_VIDEO_KEYS
                }
                for frame_dir in frame_dirs.values():
                    frame_dir.mkdir(parents=True, exist_ok=True)

                for local_frame_index in range(episode_length):
                    item = dataset[local_frame_index]
                    task = normalize_task(item.get("task"))
                    if episode_task is None:
                        episode_task = task
                        task_index = tasks_to_index.setdefault(task, len(tasks_to_index))
                    elif task != episode_task:
                        raise ValueError(
                            f"Episode {episode_index} contains multiple tasks: {episode_task!r} vs {task!r}."
                        )

                    state = as_float_vector(item[PIPER_STATE_KEY], PIPER_STATE_KEY)
                    action = as_float_vector(item[PIPER_ACTION_KEY], PIPER_ACTION_KEY)
                    base_image = to_uint8_hwc(item[PIPER_BASE_IMAGE_KEY], PIPER_BASE_IMAGE_KEY)
                    wrist_image = to_uint8_hwc(item[PIPER_WRIST_IMAGE_KEY], PIPER_WRIST_IMAGE_KEY)

                    if output_features is None:
                        output_features = build_output_features(
                            source_info["features"],
                            base_image.shape,
                            wrist_image.shape,
                            int(meta.fps),
                        )

                    actions.append(action)
                    states.append(state)
                    timestamps.append(float(np.asarray(item["timestamp"]).item()))
                    frame_indices.append(int(np.asarray(item["frame_index"]).item()))
                    task_indices.append(task_index)

                    Image.fromarray(base_image).save(frame_dirs[PIPER_BASE_IMAGE_KEY] / f"frame-{local_frame_index:06d}.png")
                    Image.fromarray(wrist_image).save(
                        frame_dirs[PIPER_WRIST_IMAGE_KEY] / f"frame-{local_frame_index:06d}.png"
                    )

                if episode_task is None or task_index is None:
                    raise ValueError(f"Episode {episode_index} is missing task metadata.")

                for video_key in REQUIRED_VIDEO_KEYS:
                    video_path = staging_dir / VIDEO_PATH_TEMPLATE.format(
                        episode_chunk=episode_chunk,
                        video_key=video_key,
                        episode_index=episode_index,
                    )
                    encode_video_frames(frame_dirs[video_key], video_path, int(meta.fps), overwrite=True)

            action_array = np.stack(actions, axis=0).astype(np.float32)
            state_array = np.stack(states, axis=0).astype(np.float32)
            timestamp_array = np.asarray(timestamps, dtype=np.float32)
            frame_index_array = np.asarray(frame_indices, dtype=np.int64)
            episode_index_array = np.full((episode_length,), episode_index, dtype=np.int64)
            index_array = np.arange(total_frames, total_frames + episode_length, dtype=np.int64)
            task_index_array = np.asarray(task_indices, dtype=np.int64)

            parquet_path = staging_dir / DATA_PATH_TEMPLATE.format(
                episode_chunk=episode_chunk,
                episode_index=episode_index,
            )
            write_episode_parquet(
                parquet_path,
                {
                    PIPER_ACTION_KEY: action_array,
                    PIPER_STATE_KEY: state_array,
                    "timestamp": timestamp_array,
                    "frame_index": frame_index_array,
                    "episode_index": episode_index_array,
                    "index": index_array,
                    "task_index": task_index_array,
                },
            )

            episode_stats = {
                PIPER_ACTION_KEY: compute_feature_stats(action_array),
                PIPER_STATE_KEY: compute_feature_stats(state_array),
                "timestamp": compute_feature_stats(timestamp_array),
                "frame_index": compute_feature_stats(frame_index_array),
                "episode_index": compute_feature_stats(episode_index_array),
                "index": compute_feature_stats(index_array),
                "task_index": compute_feature_stats(task_index_array),
            }
            all_episode_stats.append(episode_stats)
            episodes_rows.append({"episode_index": episode_index, "tasks": [episode_task], "length": episode_length})
            episodes_stats_rows.append({"episode_index": episode_index, "stats": serialize_tree(episode_stats)})
            total_frames += episode_length

        if output_features is None:
            raise ValueError(f"No frames were converted from {src}")

        stats = aggregate_stats(all_episode_stats)
        write_json(
            {
                "codebase_version": "v2.1",
                "robot_type": source_info.get("robot_type", "piper_follower"),
                "total_episodes": meta.total_episodes,
                "total_frames": total_frames,
                "total_tasks": len(tasks_to_index),
                "total_videos": meta.total_episodes * len(REQUIRED_VIDEO_KEYS),
                "total_chunks": math.ceil(meta.total_episodes / DEFAULT_CHUNK_SIZE),
                "chunks_size": DEFAULT_CHUNK_SIZE,
                "fps": int(meta.fps),
                "splits": {"train": f"0:{meta.total_episodes}"},
                "data_path": DATA_PATH_TEMPLATE,
                "video_path": VIDEO_PATH_TEMPLATE,
                "features": output_features,
            },
            staging_dir / "meta" / "info.json",
        )
        write_json(serialize_tree(stats), staging_dir / "meta" / "stats.json")
        write_jsonl(episodes_rows, staging_dir / "meta" / "episodes.jsonl")
        write_jsonl(episodes_stats_rows, staging_dir / "meta" / "episodes_stats.jsonl")
        write_jsonl(
            [
                {"task_index": task_index, "task": task}
                for task, task_index in sorted(tasks_to_index.items(), key=lambda item: item[1])
            ],
            staging_dir / "meta" / "tasks.jsonl",
        )
        write_marker(staging_dir, src, repo_id, "v3.0")
        shutil.move(staging_dir, dst)
    finally:
        shutil.rmtree(work_root, ignore_errors=True)


def load_v3_lerobot():
    repo_root = Path(__file__).resolve().parents[4]
    local_lerobot_src = repo_root / "lerobot" / "src"
    if local_lerobot_src.exists():
        sys.path.insert(0, str(local_lerobot_src))
        for module_name in list(sys.modules):
            if module_name == "lerobot" or module_name.startswith("lerobot."):
                del sys.modules[module_name]

    try:
        from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION as LEROBOT_CODEBASE_VERSION
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.datasets.video_utils import encode_video_frames
    except Exception as exc:
        raise RuntimeError(
            "Failed to import a LeRobot v3-capable loader for Piper dataset conversion. "
            "Make sure the local `lerobot/src` tree is available and its dependencies are installed."
        ) from exc

    if LEROBOT_CODEBASE_VERSION != "v3.0":
        raise RuntimeError(f"Expected LeRobot v3.0 loader, found {LEROBOT_CODEBASE_VERSION!r}.")

    return LeRobotDataset, LeRobotDatasetMetadata, encode_video_frames


def normalize_task(task: object) -> str:
    if task is None:
        raise ValueError("Episode frame is missing task metadata.")
    if isinstance(task, np.ndarray) and task.ndim == 0:
        task = task.item()
    if isinstance(task, bytes):
        task = task.decode("utf-8")
    task = str(task).strip()
    if not task:
        raise ValueError("Episode frame contains an empty task string.")
    return task


def as_float_vector(value: object, key: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (7,):
        raise ValueError(f"{key} must have shape (7,), found {array.shape}.")
    return array


def to_uint8_hwc(value: object, key: str) -> np.ndarray:
    image = np.asarray(value)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"{key} must have 3 dimensions, found shape {image.shape}.")
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] != 3:
        raise ValueError(f"{key} must have 3 channels, found shape {image.shape}.")
    return image.astype(np.uint8, copy=False)


def build_output_features(source_features: dict, base_shape: tuple[int, int, int], wrist_shape: tuple[int, int, int], fps: int) -> dict:
    return {
        PIPER_ACTION_KEY: {
            "dtype": "float32",
            "shape": list(source_features[PIPER_ACTION_KEY]["shape"]),
            "names": source_features[PIPER_ACTION_KEY].get("names"),
        },
        PIPER_STATE_KEY: {
            "dtype": "float32",
            "shape": list(source_features[PIPER_STATE_KEY]["shape"]),
            "names": source_features[PIPER_STATE_KEY].get("names"),
        },
        PIPER_BASE_IMAGE_KEY: make_video_feature(source_features.get(PIPER_BASE_IMAGE_KEY, {}), base_shape, fps),
        PIPER_WRIST_IMAGE_KEY: make_video_feature(source_features.get(PIPER_WRIST_IMAGE_KEY, {}), wrist_shape, fps),
        **DEFAULT_FEATURES,
    }


def make_video_feature(source_feature: dict, image_shape: tuple[int, int, int], fps: int) -> dict:
    height, width, channels = image_shape
    info = dict(source_feature.get("info", {}))
    info.update(
        {
            "video.height": height,
            "video.width": width,
            "video.codec": info.get("video.codec", "av1"),
            "video.pix_fmt": info.get("video.pix_fmt", "yuv420p"),
            "video.is_depth_map": info.get("video.is_depth_map", False),
            "video.fps": fps,
            "video.channels": channels,
            "has_audio": info.get("has_audio", False),
        }
    )
    return {
        "dtype": "video",
        "shape": [height, width, channels],
        "names": source_feature.get("names", ["height", "width", "channels"]),
        "info": info,
    }


def compute_feature_stats(array: np.ndarray) -> dict[str, np.ndarray]:
    array = np.asarray(array)
    keepdims = array.ndim == 1
    return {
        "min": np.min(array, axis=0, keepdims=keepdims),
        "max": np.max(array, axis=0, keepdims=keepdims),
        "mean": np.mean(array, axis=0, keepdims=keepdims),
        "std": np.std(array, axis=0, keepdims=keepdims),
        "count": np.asarray([len(array)], dtype=np.int64),
    }


def aggregate_stats(stats_list: list[dict[str, dict[str, np.ndarray]]]) -> dict[str, dict[str, np.ndarray]]:
    aggregated: dict[str, dict[str, np.ndarray]] = {}
    for key in {feature_key for stats in stats_list for feature_key in stats}:
        aggregated[key] = aggregate_feature_stats([stats[key] for stats in stats_list if key in stats])
    return aggregated


def aggregate_feature_stats(feature_stats_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    means = np.stack([stats["mean"] for stats in feature_stats_list])
    variances = np.stack([stats["std"] ** 2 for stats in feature_stats_list])
    counts = np.stack([stats["count"] for stats in feature_stats_list])
    total_count = counts.sum(axis=0)

    expanded_counts = counts
    while expanded_counts.ndim < means.ndim:
        expanded_counts = np.expand_dims(expanded_counts, axis=-1)

    total_mean = (means * expanded_counts).sum(axis=0) / total_count
    mean_delta = means - total_mean
    total_variance = ((variances + mean_delta**2) * expanded_counts).sum(axis=0) / total_count

    return {
        "min": np.min(np.stack([stats["min"] for stats in feature_stats_list]), axis=0),
        "max": np.max(np.stack([stats["max"] for stats in feature_stats_list]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": total_count,
    }


def serialize_tree(tree):
    if isinstance(tree, dict):
        return {key: serialize_tree(value) for key, value in tree.items()}
    if isinstance(tree, np.ndarray):
        return tree.tolist()
    if isinstance(tree, np.generic):
        return tree.item()
    return tree


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4))


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, mode="w") as writer:
        writer.write_all(rows)


def write_episode_parquet(path: Path, columns: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            PIPER_ACTION_KEY: pa.array(columns[PIPER_ACTION_KEY].tolist(), type=pa.list_(pa.float32())),
            PIPER_STATE_KEY: pa.array(columns[PIPER_STATE_KEY].tolist(), type=pa.list_(pa.float32())),
            "timestamp": pa.array(columns["timestamp"], type=pa.float32()),
            "frame_index": pa.array(columns["frame_index"], type=pa.int64()),
            "episode_index": pa.array(columns["episode_index"], type=pa.int64()),
            "index": pa.array(columns["index"], type=pa.int64()),
            "task_index": pa.array(columns["task_index"], type=pa.int64()),
        }
    )
    pq.write_table(table, path)


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
