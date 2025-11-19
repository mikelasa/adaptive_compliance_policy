from __future__ import annotations

from typing import Any, Dict, Optional

import requests
import numpy as np


class MPCClient:
    """
    Client to talk to the MPC controller server (running on the GPU machine).

    It exposes two main operations:
      - set_observation(obs_task)
      - compute_sparse_control(time_now)

    Parameters
    ----------
    base_url : str
        Base URL of the server, e.g. "http://192.168.0.10:8000"
        (without trailing slash).
    timeout : float
        Default timeout (in seconds) for HTTP requests.
    """

    def __init__(self, base_url: str, timeout: float = 5.0) -> None:
        # remove trailing slash if present
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    # ---------- internal helpers ----------

    @staticmethod
    def _to_serializable(x: Any) -> Any:
        """
        Convert numpy arrays / scalars to plain Python types
        so they can be JSON-encoded.
        """
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.float32, np.float64)):
            return float(x)
        if isinstance(x, (np.int32, np.int64)):
            return int(x)
        if isinstance(x, dict):
            return {k: MPCClient._to_serializable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [MPCClient._to_serializable(v) for v in x]
        return x

    def _post(self, path: str, json_body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal helper to send a POST request and return parsed JSON.
        Raises requests.HTTPError on non-2xx.
        """
        url = f"{self.base_url}{path}"
        resp = self.session.post(url, json=json_body, timeout=self.timeout)
        resp.raise_for_status()
        # FastAPI returns JSON; parse it into a dict
        return resp.json()

    # ---------- public API ----------

    def set_observation(self, obs_task: Dict[str, Any]) -> None:
        """
        Send observation to the MPC server.

        Parameters
        ----------
        obs_task : dict
            The same dict you would pass to controller.set_observation(...)
            on the server side, typically obs_task["obs"] from your runner.

            Keys: strings
            Values: lists or numpy arrays (timestamps, low-dim obs, etc.)
        """
        serializable_obs = self._to_serializable(obs_task)
        payload = {"obs_task": serializable_obs}
        _ = self._post("/set_observation", payload)
        # If you want, you can inspect the response status / message.
        # For now, we just ignore it unless there is an HTTP error.

    def compute_sparse_control(
        self,
        time_now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Request sparse control from the MPC server.

        Parameters
        ----------
        time_now : float, optional
            Current time (in the same time base as your timestamps), e.g.
            env.current_hardware_time_s. If None, the server can skip lag logs.

        Returns
        -------
        result : dict
            For pose9:
                { "action": ... }
            For pose9pose9s1:
                {
                    "target_mats": ...,
                    "vt_mats": ...,
                    "stiffness": ...
                }
            The exact structure matches what the server returns.
        """
        payload = {"time_now": time_now}
        result = self._post("/compute_sparse_control", payload)
        return result
    
    def get_shape_meta(self) -> Dict[str, Any]:
        return self._post("/shape_meta", {})

