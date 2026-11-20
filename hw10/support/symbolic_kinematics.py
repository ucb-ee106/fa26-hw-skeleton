"""Provided symbolic kinematics for the UR5's unit revolute joints."""

import sympy as sp


def _hat(v):
    x, y, z = v
    return sp.Matrix([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def _joint_transform(twist, angle):
    v, w = twist[:3, :], twist[3:, :]
    W = _hat(w)
    R = sp.eye(3) + sp.sin(angle) * W + (1 - sp.cos(angle)) * W**2
    p = (sp.eye(3) - R) @ w.cross(v)
    return R.row_join(p).col_join(sp.Matrix([[0, 0, 0, 1]]))


def forward_kinematics(twists, theta):
    """Return the SymPy (4, 4) product of the joint transformations.

    twists is a SymPy (6, N) matrix of home twists and theta contains
    the N joint angles. Multiply the result by the link's home pose
    to obtain its pose at theta.
    """
    pose = sp.eye(4)
    for i, angle in enumerate(theta):
        pose = pose @ _joint_transform(twists[:, i], angle)
    return pose


def adjoint(pose):
    """Return the SymPy (6, 6) adjoint of a SymPy (4, 4) pose.

    The adjoint maps twists from the local frame into the frame in which
    pose is expressed. Use the inverse pose to map in the other direction.
    """
    R, p = pose[:3, :3], pose[:3, 3]
    return R.row_join(_hat(p) @ R).col_join(sp.zeros(3, 3).row_join(R))
