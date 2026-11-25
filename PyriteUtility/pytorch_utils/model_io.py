import sys
import os
from typing import Dict, Callable, Tuple, List

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, "../../"))

import torch
import dill
import hydra
from omegaconf import DictConfig, OmegaConf

# TODO: Remove this dependency
from PyriteML.diffusion_policy.workspace.base_workspace import BaseWorkspace
from PyriteML.diffusion_policy.workspace.train_diffusion_unet_image_workspace import (
    TrainDiffusionUnetImageWorkspace,
)


def load_policy_and_cfg(ckpt_path, device):
    """Load a policy and return it together with the whole checkpoint config.

    Same loading path as load_policy(), which is a thin wrapper over this.
    Callers that need more than task.shape_meta use this one -- a vision-only
    policy, for instance, needs cfg.shape_meta as well, because that is the
    reduced meta the policy network actually sees (vision + proprioception in,
    pose9 out) while cfg.task.shape_meta is still the full ACP meta that drives
    the sampler and the relative-pose conversion.
    """
    # load checkpoint
    if not ckpt_path.endswith(".ckpt"):
        ckpt_path = os.path.join(ckpt_path, "checkpoints", "latest.ckpt")
    payload = torch.load(open(ckpt_path, "rb"), map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    # print("model_name:", cfg.policy.obs_encoder.model_name)
    # The dataset path sits in different places depending on the task config: a
    # plain ACP dataset puts it on task.dataset, while a wrapped one (e.g.
    # DPAdapterDataset, whose inner dataset is the ACP one) puts it a level
    # deeper. Select rather than attribute-access so a layout we have not seen
    # prints "unknown" instead of raising during policy load.
    dataset_path = (
        OmegaConf.select(cfg, "task.dataset.dataset_path", default=None)
        or OmegaConf.select(cfg, "task.dataset.inner.dataset_path", default=None)
        or OmegaConf.select(cfg, "task.dataset_path", default=None)
        or "unknown"
    )
    print("dataset_path:", dataset_path)

    cls = hydra.utils.get_class(cfg._target_)
    # TODO(Yifan) Get policy independent from workspace
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.num_inference_steps = (
        cfg.policy.num_inference_steps
    )  # DDIM inference iterations

    policy.eval().to(device)
    policy.reset()
    return policy, cfg


def load_policy(ckpt_path, device):
    policy, cfg = load_policy_and_cfg(ckpt_path, device)
    return policy, cfg.task.shape_meta


def serialize_model(ckpt_path):
    policy, shape_meta = load_policy(ckpt_path, "cuda")
    sm = torch.jit.script(policy)
    sm.save(ckpt_path.replace(".ckpt", ".pt"))


# testing
class MyModule(torch.nn.Module):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

    def forward(self, input):
        if input.sum() > 0:
            output = self.weight.mv(input)
        else:
            output = self.weight + input
        return output

    def hahaha(self, x):
        x = x + 1
        return x


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--ckpt_path", type=str)
    # args = parser.parse_args()
    # serialize_model(args.ckpt_path)

    my_module = MyModule(10, 20)
    sm = torch.jit.script(my_module)
    sm.save("/tmp/test.pt")
    print("done")
