"""Adapter that exposes the ACP dataset in the format vanilla Diffusion Policy expects.

The ACP data pipeline (VirtualTargetDataset + SequenceSampler) emits a nested,
multi-clock batch:

    {"obs": {"sparse": {rgb_0, robot0_eef_pos, robot0_eef_rot_axis_angle,
                        robot0_eef_wrench, *_time_stamps_*}},
     "action": {"sparse": (T, 19 * n_arms)}}

Upstream Diffusion Policy expects a flat batch with no force channel and a plain
pose action:

    {"obs": {rgb_0, robot0_eef_pos, robot0_eef_rot_axis_angle},
     "action": (T, 9 * n_arms)}

This class does that translation and nothing else, so the DP baseline trains on
exactly the same demonstrations, the same episode train/val split and the same
relative-pose conventions as ACP -- the only differences being the ones that
define the baseline: no wrench input, and no virtual target / stiffness output.

The 19-dim ACP action is laid out per arm as
    [0:9]   pose9 nominal command, relative to the first action frame
    [9:18]  pose9 virtual target, same frame
    [18]    stiffness scalar
so slicing [19*i : 19*i+9] is already the nominal pose command a position
controller would track. No change to the ACP action pipeline is needed.
"""

from typing import Dict, List, Optional, Union

import copy

import torch
from omegaconf import DictConfig, OmegaConf
import hydra

from diffusion_policy.dataset.base_dataset import BaseDataset
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)


class DPAdapterDataset(BaseDataset):
    def __init__(
        self,
        inner: Union[BaseDataset, DictConfig],
        policy_shape_meta: dict,
        n_obs_steps: int,
        action_dim_per_arm: int = 19,
        action_keep_dims: int = 9,
    ):
        """
        inner: the ACP dataset (hydra instantiates it from the nested config).
        policy_shape_meta: the shape_meta handed to the DP policy / encoder. Its
            "obs" keys select which of the inner dataset's obs keys survive --
            wrench and timestamp keys are simply left out of it.
        n_obs_steps: obs sequences are truncated to their LAST n_obs_steps, so
            every key reaches the encoder with the horizon the policy expects
            regardless of the inner dataset's per-key horizons, and the most
            recent frame is always included. The inner horizons are left at
            ACP's values on purpose: the SequenceSampler derives its valid
            start indices from them, so keeping them identical means the DP
            baseline trains on exactly the same samples, in the same order,
            with the same train/val episode split as ACP.
        """
        super().__init__()
        if isinstance(inner, (DictConfig, dict)):
            inner = hydra.utils.instantiate(inner)
        self.inner = inner

        if isinstance(policy_shape_meta, DictConfig):
            policy_shape_meta = OmegaConf.to_container(policy_shape_meta, resolve=True)
        self.policy_shape_meta = policy_shape_meta
        self.n_obs_steps = n_obs_steps
        self.action_dim_per_arm = action_dim_per_arm
        self.action_keep_dims = action_keep_dims

        self.obs_keys: List[str] = list(policy_shape_meta["obs"].keys())
        self.id_list = list(inner.id_list)

        # which columns of the 19-dim-per-arm action we keep
        self.action_index = torch.cat(
            [
                torch.arange(
                    i * action_dim_per_arm,
                    i * action_dim_per_arm + action_keep_dims,
                    dtype=torch.long,
                )
                for i in range(len(self.id_list))
            ]
        )
        expected = policy_shape_meta["action"]["shape"][0]
        assert len(self.action_index) == expected, (
            f"policy_shape_meta.action.shape is {expected} but slicing "
            f"{action_keep_dims} dims from each of {len(self.id_list)} arms "
            f"gives {len(self.action_index)}"
        )

    # ========= dataset interface ============
    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.inner[idx]
        obs_sparse = sample["obs"]["sparse"]

        obs = dict()
        for key in self.obs_keys:
            value = obs_sparse[key]
            assert value.shape[0] >= self.n_obs_steps, (
                f"obs key {key} has horizon {value.shape[0]} in the inner "
                f"dataset, which is shorter than n_obs_steps={self.n_obs_steps}"
            )
            # the sampler returns frames in chronological order, so the most
            # recent observation is last -- take the tail, not the head
            obs[key] = value[-self.n_obs_steps :]

        action = sample["action"]["sparse"][..., self.action_index]
        return {"obs": obs, "action": action}

    def get_validation_dataset(self) -> "DPAdapterDataset":
        val_set = copy.copy(self)
        val_set.inner = self.inner.get_validation_dataset()
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        """Reuse the inner dataset's statistics, then drop/slice to DP's keys.

        The stats are identical to the ones ACP trains against (same range
        normalizer on position, identity on rotation_6d), so the two models see
        the same targets in the same units.
        """
        inner_normalizer = self.inner.get_normalizer(**kwargs)

        normalizer = LinearNormalizer()
        for key in self.obs_keys:
            assert key in inner_normalizer.params_dict, (
                f"obs key {key} has no normalizer in the inner dataset"
            )
            normalizer[key] = inner_normalizer[key]
        normalizer["action"] = self._slice_action_normalizer(
            inner_normalizer["action"]
        )
        return normalizer

    def _slice_action_normalizer(
        self, src: SingleFieldLinearNormalizer
    ) -> SingleFieldLinearNormalizer:
        params = src.params_dict
        index = self.action_index.to(params["scale"].device)
        return SingleFieldLinearNormalizer.create_manual(
            scale=params["scale"][..., index],
            offset=params["offset"][..., index],
            input_stats_dict={
                k: v[..., index] for k, v in params["input_stats"].items()
            },
        )
