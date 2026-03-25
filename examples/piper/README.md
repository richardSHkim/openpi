# Piper

This example adds a native Piper fine-tuning and serving path for `openpi` using the existing `pi05` best-practice flow:

1. Stage your dataset as a LeRobot `v2.1` dataset that `third_party/openpi` can read
2. Compute fresh normalization statistics
3. Fine-tune `pi05` on the staged dataset
4. Serve the checkpoint with `scripts/serve_policy.py`
5. Smoke-test the server with `examples/simple_client/main.py --env PIPER`

## 1. Convert Or Stage The Dataset

`third_party/openpi` currently pins LeRobot `v2.1`, while `datasets/richardshkim/piper_banana_v2` is expected to be a `v3.0` dataset. The converter below handles both cases:

- `v2.1` input: validate and copy to `$HF_LEROBOT_HOME/<repo_id>`
- `v3.0` input: read episode-by-episode with a `v3.0` loader and rewrite a `v2.1` staging dataset

```bash
uv run examples/piper/convert_piper_dataset.py \
  --src /path/to/datasets/richardshkim/piper_banana_v2 \
  --repo-id richardshkim/piper_banana_v2
```

If your current local source is already a `v2.1` dataset, you can still keep the canonical training repo id:

```bash
uv run examples/piper/convert_piper_dataset.py \
  --src /path/to/datasets/richardshkim/piper_banana_v2_v2.1 \
  --repo-id richardshkim/piper_banana_v2
```

By default, the staged dataset is written to:

```bash
$HF_LEROBOT_HOME/richardshkim/piper_banana_v2
```

Use `--dst-root /path/to/lerobot_home` if you want a different staging root.

## 2. Compute Norm Stats

Piper is not one of the robots that ships with precomputed openpi normalization assets, so compute fresh stats from the staged dataset:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_piper
```

Thin wrapper:

```bash
bash examples/piper/compute_norm_stats.sh
```

## 3. Train

`pi05_piper` is registered in [`src/openpi/training/config.py`](../../src/openpi/training/config.py) and points at the staged dataset `richardshkim/piper_banana_v2`.

Thin wrappers:

```bash
bash examples/piper/prepare_dataset.sh
bash examples/piper/compute_norm_stats.sh
bash examples/piper/train.sh
```

The wrappers intentionally stay very small and map almost 1:1 to the Python entrypoints.
Useful overrides:

```bash
PIPER_EXP_NAME=my_experiment \
PIPER_DATASET_SRC=/path/to/datasets/richardshkim/piper_banana_v2 \
PIPER_FSDP_DEVICES=2 \
bash examples/piper/prepare_dataset.sh

PIPER_EXP_NAME=my_experiment \
PIPER_FSDP_DEVICES=2 \
bash examples/piper/train.sh --batch-size 2
```

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_piper --exp-name=my_experiment --overwrite
```

For a short smoke test:

```bash
uv run scripts/train.py \
  pi05_piper \
  --exp-name=smoke \
  --batch-size=2 \
  --num-train-steps=1 \
  --save-interval=1 \
  --log-interval=1 \
  --overwrite
```

## 4. Serve

Use checkpoint mode, which is the recommended openpi path for custom embodiments:

```bash
uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi05_piper \
  --policy.dir=checkpoints/pi05_piper/my_experiment/<step>
```

Thin wrapper:

```bash
PIPER_POLICY_DIR=checkpoints/pi05_piper/my_experiment/<step> \
bash examples/piper/serve.sh
```

The Piper policy expects inference payloads shaped like:

```python
{
    "images/base": ...,
    "images/wrist": ...,
    "state": ...,
    "prompt": ...,
}
```

## 5. Smoke-Test The Server

In a second terminal:

```bash
uv run examples/simple_client/main.py --env PIPER --host localhost --port 8000 --num-steps 2
```

That path uses [`src/openpi/policies/piper_policy.py`](../../src/openpi/policies/piper_policy.py) and should return action chunks with shape `(action_horizon, 7)`.
