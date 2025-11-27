# model_server.py
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.append(os.path.join(sys.path[0], ".."))

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, "../../"))

# Import your refactored controller class
from PyriteUtility.planning_control.mpc_refactorized import ModelPredictiveControllerHybrid  
from PyriteUtility.pytorch_utils.model_io import load_policy
from PyriteEnvSuites.utils.env_utils import pose9pose9s1_to_traj
from omegaconf import OmegaConf

checkpoint_folder_path = os.environ.get("PYRITE_CHECKPOINT_FOLDERS")

policy_para = {
        "save_low_dim_every_N_frame": 1,
        "save_visual_every_N_frame": 1,
        "ckpt_path": "/2025.11.04_10.05.05_Wipe_single_arm_Wipe_single_arm/checkpoints/latest.ckpt",
    }

# ---------------------------------------------------------------------
# CONFIG: device, shape_meta, policy loading, action_to_trajectory
# ---------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global controller instance
controller: Optional[ModelPredictiveControllerHybrid] = None

def init_controller() -> ModelPredictiveControllerHybrid:
    # TODO: choose correct id_list (e.g. [0] or [0, 1] depending on your setup)
    id_list = [0]
    print("Loading policy: ", checkpoint_folder_path + policy_para["ckpt_path"])
    device = torch.device(DEVICE)
    policy, shape_meta = load_policy(
        checkpoint_folder_path + policy_para["ckpt_path"], device
    )

    action_to_trajectory = pose9pose9s1_to_traj

    ctrl = ModelPredictiveControllerHybrid(
        shape_meta=shape_meta,
        id_list=id_list,
        policy=policy,
        action_to_trajectory=action_to_trajectory,
        sparse_execution_horizon=10,
    )
    # convenience: store device inside controller
    ctrl._device = DEVICE
    return ctrl


# ---------------------------------------------------------------------
# FastAPI app & models
# ---------------------------------------------------------------------

app = FastAPI(title="MPC Model Server")

class ObservationRequest(BaseModel):
    obs_task: Dict[str, Any]


class ComputeSparseControlRequest(BaseModel):
    time_now: Optional[float] = None  # same time base as obs timestamps


@app.on_event("startup")
async def startup_event():
    global controller
    controller = init_controller()
    print("[model_server] Controller initialized on device:", DEVICE)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _ensure_controller() -> ModelPredictiveControllerHybrid:
    if controller is None:
        raise HTTPException(status_code=500, detail="Controller not initialized")
    return controller


# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------

@app.post("/set_observation")
def set_observation(req: ObservationRequest):
    ctrl = _ensure_controller()

    # You can keep values as lists; your original set_observation
    # only uses len() and slicing, which works on lists.
    # If you prefer numpy, uncomment the conversion:
    obs_task_np = {}
    for k, v in req.obs_task.items():
        if isinstance(v, list):
            obs_task_np[k] = np.array(v)
        else:
            obs_task_np[k] = v

    ctrl.set_observation(obs_task_np)
    return {"status": "ok"}


@app.post("/compute_sparse_control")
def compute_sparse_control(req: ComputeSparseControlRequest):
    ctrl = _ensure_controller()

    # Call your controller's compute_sparse_control.
    # We assume you've refactored it to: compute_sparse_control(device, time_now=None)
    action = ctrl.compute_sparse_control(ctrl._device, time_now=req.time_now)

    # For pose9: action is a single array
    # For pose9pose9s1: action is a tuple (target_mats, vt_mats, stiffness)
    if isinstance(action, tuple):
        target_mats, vt_mats, stiffness = action
        return {
            "target_mats": np.asarray(target_mats).tolist(),
            "vt_mats": np.asarray(vt_mats).tolist(),
            "stiffness": np.asarray(stiffness).tolist(),
        }
    else:
        return {
            "action": np.asarray(action).tolist()
        }


# Optional: if you also want an endpoint for update_control:
class UpdateControlRequest(BaseModel):
    time_step: float
    time_now: Optional[float] = None


@app.post("/update_control")
def update_control(req: UpdateControlRequest):
    ctrl = _ensure_controller()

    target = ctrl.update_control(
        time_step=req.time_step,
        device=ctrl._device,
        time_now=req.time_now,
    )

    if target is None:
        return {"status": "new_horizon"}
    else:
        return {"target": np.asarray(target).tolist()}
    
# inside model_server.py
@app.get("/shape_meta")
def get_shape_meta():
    ctrl = _ensure_controller()
    # Convert OmegaConf DictConfig/ListConfig to plain dict/list
    plain = OmegaConf.to_container(ctrl.shape_meta, resolve=True)
    return plain



# ---------------------------------------------------------------------
# Entry point (so you can run: python model_server.py)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("model_server:app", host="0.0.0.0", port=8000, reload=True)
