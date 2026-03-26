import numpy as np

import openpi.models.model as _model
import openpi.policies.piper_policy as piper_policy
from openpi.training import config as _config
import openpi.transforms as _transforms


def test_piper_inputs_chw_float():
    transform = piper_policy.PiperInputs(model_type=_model.ModelType.PI05)

    base_image = np.full((3, 4, 5), 0.5, dtype=np.float32)
    wrist_image = np.full((3, 4, 5), 0.25, dtype=np.float32)
    state = np.arange(7, dtype=np.float32)
    actions = np.arange(14, dtype=np.float32).reshape(2, 7)

    outputs = transform(
        {
            "images/base": base_image,
            "images/wrist": wrist_image,
            "state": state,
            "actions": actions,
            "prompt": b"pick banana",
        }
    )

    np.testing.assert_array_equal(outputs["state"], state)
    np.testing.assert_array_equal(outputs["actions"], actions)
    np.testing.assert_array_equal(outputs["image"]["base_0_rgb"], np.full((4, 5, 3), 127, dtype=np.uint8))
    np.testing.assert_array_equal(outputs["image"]["left_wrist_0_rgb"], np.full((4, 5, 3), 63, dtype=np.uint8))
    np.testing.assert_array_equal(outputs["image"]["right_wrist_0_rgb"], np.zeros((4, 5, 3), dtype=np.uint8))
    assert bool(outputs["image_mask"]["base_0_rgb"])
    assert bool(outputs["image_mask"]["left_wrist_0_rgb"])
    assert not bool(outputs["image_mask"]["right_wrist_0_rgb"])
    assert outputs["prompt"] == "pick banana"


def test_piper_inputs_hwc_uint8_fast_model():
    transform = piper_policy.PiperInputs(model_type=_model.ModelType.PI0_FAST)

    base_image = np.random.randint(256, size=(8, 6, 3), dtype=np.uint8)
    wrist_image = np.random.randint(256, size=(8, 6, 3), dtype=np.uint8)
    state = np.arange(7, dtype=np.float32)

    outputs = transform(
        {
            "images": {
                "base": base_image,
                "wrist": wrist_image,
            },
            "state": state,
        }
    )

    np.testing.assert_array_equal(outputs["state"], state)
    np.testing.assert_array_equal(outputs["image"]["base_0_rgb"], base_image)
    np.testing.assert_array_equal(outputs["image"]["base_1_rgb"], np.zeros_like(base_image))
    np.testing.assert_array_equal(outputs["image"]["wrist_0_rgb"], wrist_image)
    assert bool(outputs["image_mask"]["base_0_rgb"])
    assert bool(outputs["image_mask"]["base_1_rgb"])
    assert bool(outputs["image_mask"]["wrist_0_rgb"])


def test_piper_outputs_slice_first_seven_dims():
    transform = piper_policy.PiperOutputs()
    model_actions = np.arange(64, dtype=np.float32).reshape(2, 32)

    outputs = transform({"actions": model_actions})

    np.testing.assert_array_equal(outputs["actions"], model_actions[:, :7])


def test_piper_delta_absolute_round_trip():
    transform = piper_policy.PiperInputs(model_type=_model.ModelType.PI05)
    delta_mask = _transforms.make_bool_mask(6, -1)

    inputs = transform(
        {
            "images/base": np.zeros((8, 8, 3), dtype=np.uint8),
            "images/wrist": np.zeros((8, 8, 3), dtype=np.uint8),
            "state": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.25], dtype=np.float32),
            "actions": np.array(
                [
                    [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 0.9],
                    [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 0.1],
                ],
                dtype=np.float32,
            ),
        }
    )

    original_actions = inputs["actions"].copy()
    delta_outputs = _transforms.DeltaActions(delta_mask)(
        {"state": inputs["state"].copy(), "actions": inputs["actions"].copy()}
    )

    np.testing.assert_allclose(
        delta_outputs["actions"],
        np.array(
            [
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.9],
                [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.1],
            ],
            dtype=np.float32,
        ),
    )

    absolute_outputs = _transforms.AbsoluteActions(delta_mask)(
        {"state": inputs["state"].copy(), "actions": delta_outputs["actions"].copy()}
    )
    np.testing.assert_allclose(absolute_outputs["actions"], original_actions)


def test_pi05_piper_config_registration():
    config = _config.get_config("pi05_piper")
    data_config = config.data.create(config.assets_dirs, config.model)

    assert data_config.repo_id == "richardshkim/piper_banana_v2_openpi"
    assert data_config.action_sequence_keys == ("action",)
    assert isinstance(data_config.data_transforms.inputs[0], piper_policy.PiperInputs)
