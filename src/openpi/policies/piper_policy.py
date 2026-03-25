import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_piper_example() -> dict:
    """Creates a random input example for the Piper policy."""
    return {
        "images/base": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "images/wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "state": np.random.rand(7).astype(np.float32),
        "prompt": "put banana into basket",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(255 * image, 0, 255).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"Expected image to have 3 dimensions, got shape {image.shape}.")
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        flat_data = transforms.flatten_dict(data)
        base_image = _parse_image(flat_data["images/base"])
        wrist_image = _parse_image(flat_data["images/wrist"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                image_names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                image_names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": np.asarray(flat_data["state"], dtype=np.float32),
            "image": dict(zip(image_names, images, strict=True)),
            "image_mask": dict(zip(image_names, image_masks, strict=True)),
        }

        if "actions" in flat_data:
            inputs["actions"] = np.asarray(flat_data["actions"], dtype=np.float32)

        if "prompt" in flat_data:
            prompt = flat_data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}
