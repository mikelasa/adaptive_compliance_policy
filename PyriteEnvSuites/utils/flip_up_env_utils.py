import numpy as np
from spatialmath.base import q2r

def is_flip_up_success(pose7d_obj, angle_deg_thresh):
    unit_z = np.array([0, 0, 1])
    q = pose7d_obj[3:]
    r = q2r(q)
    x = r[:, 0]

    angle_deg = 180/np.pi * np.arccos(np.dot(x, unit_z))

    return angle_deg < angle_deg_thresh, angle_deg