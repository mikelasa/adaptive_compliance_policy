import numpy as np
import spatialmath as sm
import spatialmath.base as smb
from PyriteUtility.planning_control.trajectory import (
    LinearInterpolator,
    LinearTransformationInterpolator,
    CombinedGeometricPath,
)


def ts_to_js_traj(action_mats, time_steps, robot):
    """
    Convert an array of task space waypoints to a joint space trajectory.
    Args:
        action_mats: (T, 4, 4) array of task space waypoints.
        time_steps: (T,) array of time points for each waypoint.
        robot: MujocoRobot object with inverse_kinematics_SE3() method.
    Returns:
        A spline trajectory in joint space
    A ValueError is raised if no IK solution is found.
    """
    assert action_mats.shape[1] == 4
    assert action_mats.shape[2] == 4
    if action_mats.shape[0] != time_steps.shape[0]:
        print("action_mats.shape[0]:", action_mats.shape[0])
        print("time_steps.shape[0]:", time_steps.shape[0])
        raise ValueError(
            "The number of time steps must match the number of action matrices."
        )
    jpos_waypoints = []
    for mat in action_mats:
        q = smb.r2q(mat[:3, :3], check=False)
        q = q / np.linalg.norm(q)
        pose7 = np.concatenate([mat[:3, 3], q])
        ik_result = robot.inverse_kinematics(pose7, True)
        if ik_result is None:
            raise ValueError("No IK solution found.")
        jpos_waypoints.append(ik_result)
    return LinearInterpolator(time_steps, jpos_waypoints)


def pose9pose9s1_to_traj(target_mats, vt_mats, stiffness, time_steps):
    """
    Args:
        target_mats: (T, 4, 4) array of task space waypoints.
        vt_mats: (T, 4, 4) array of task space virtual target waypoints.
        stiffness: (T,) array of stiffness for each waypoint.
        time_steps: (T,) array of time points for each waypoint.
    Returns:
        A trajectory with concatenated 19-dimensional waypoints.
    A ValueError is raised if no IK solution is found.
    """
    assert target_mats.shape[1] == 4
    assert target_mats.shape[2] == 4
    assert vt_mats.shape[1] == 4
    assert vt_mats.shape[2] == 4
    if (
        target_mats.shape[0] != time_steps.shape[0]
        or vt_mats.shape[0] != time_steps.shape[0]
        or stiffness.shape[0] != time_steps.shape[0]
    ):
        print("target_mats.shape[0]:", target_mats.shape[0])
        print("vt_mats.shape[0]:", vt_mats.shape[0])
        print("stiffness.shape[0]:", stiffness.shape[0])
        print("time_steps.shape[0]:", time_steps.shape[0])
        raise ValueError("The number of time steps must match among sources of inputs.")

    target_traj = LinearTransformationInterpolator(time_steps, target_mats)
    vt_traj = LinearTransformationInterpolator(time_steps, vt_mats)
    stiffness_traj = LinearInterpolator(time_steps, stiffness)

    return CombinedGeometricPath([target_traj, vt_traj, stiffness_traj])
