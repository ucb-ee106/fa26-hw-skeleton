"""HW10: UR5 dynamics and torque control.

Use the provided SymPy forward_kinematics and adjoint helpers below.
"""

import numpy as np
import sympy as sp

from support.symbolic_kinematics import adjoint, forward_kinematics


def link_jacobian(theta, twists, home_pose, link_number):
    """Return the body Jacobian of a link's center-of-mass frame.

    Args:
        theta: Six SymPy joint angle symbols, in radians.
        twists: SymPy (6, 6) matrix of home joint twists in the base frame.
        home_pose: SymPy (4, 4) home pose of this link's center of mass.
        link_number: Link number from 1 to 6, counted from the base.

    Returns:
        SymPy (6, 6) matrix.
    """
    # TODO
    raise NotImplementedError


def inertia_matrix(theta, twists, home_poses, link_inertias):
    """Build the symbolic joint-space inertia matrix M(theta).

    Args:
        theta: Sequence of six SymPy joint angle symbols, in radians.
        twists: SymPy (6, 6) matrix of home joint twists in the base frame.
        home_poses: Six SymPy (4, 4) center-of-mass home poses.
        link_inertias: Six SymPy (6, 6) inertias.

    Returns:
        SymPy matrix (6, 6).
        Use link_jacobian to get each link's body Jacobian.
    """
    # TODO (a)
    raise NotImplementedError


def coriolis_matrix(M, q, qd):
    """Build the symbolic Coriolis and centripetal matrix.

    Args:
        M: SymPy mass matrix (6, 6), expressed in q.
        q: Six SymPy symbols for joint positions in radians.
        qd: Six SymPy symbols for joint velocities in rad/s.

    Returns:
        SymPy matrix C (6, 6), using the Christoffel formula.


    Hint, use M.diff(angle) to take the partial derivative of M
    with respect to a specific angle with sympy.
    """
    # TODO (a)
    raise NotImplementedError


def gravity_vector(theta, twists, home_poses, link_inertias, gravity):
    """Build the symbolic gravity compensation torque G(theta).

    Args:
        theta: Sequence of six SymPy joint angle symbols, in radians.
        twists: SymPy (6, 6) matrix of home joint twists in the base frame.
        home_poses: Six SymPy (4, 4) center-of-mass home poses.
        link_inertias: Six SymPy (6, 6) generalized inertias in the COM frames.
        gravity: SymPy (3, 1) base-frame gravity vector, in m/s^2.

    Returns:
        SymPy column (6, 1) of gravity compensation torques in N m.
    """
    # TODO (a)
    raise NotImplementedError


def forward_dynamics(q, qd, tau, dynamics):
    """Return joint acceleration (6,) in rad/s^2 for the supplied plant.

    q, qd and tau are NumPy arrays (6,) in radians, rad/s and N m.
    dynamics(q, qd) returns M (6, 6), C (6, 6) and G (6,).
    """
    M, C, G = dynamics(q, qd)
    return np.linalg.solve(M, tau - C @ qd - G)


def integrate_step(q, qd, tau, dt, dynamics):
    """Advance one forward Euler step.

    q, qd and tau are NumPy arrays (6,) in radians, rad/s and N m.
    dt is the timestep in seconds, and dynamics is the plant callable.
    Return (q_next, qd_next), two arrays (6,) in radians and rad/s.
    Both increments are evaluated at the input state.
    """
    qdd = forward_dynamics(q, qd, tau, dynamics)
    return q + dt * qd, qd + dt * qdd


def computed_torque(q, qd, q_ref, qd_ref, qdd_ref, kp, kd, dynamics):
    """Compute joint torques with feedback in acceleration units.

    Args:
        q, qd: Measured NumPy arrays (6,) in radians and rad/s.
        q_ref, qd_ref, qdd_ref: Desired NumPy arrays (6,) in radians,
            rad/s, and rad/s^2.
        kp, kd: NumPy arrays (6,) of per-joint gains in 1/s^2 and 1/s.
        dynamics: Nominal model function callable as described in forward_dynamics.
            Can be called as dynamics(q, qd)

    Returns:
        NumPy array (6,) of joint torques in N m.
    """
    # TODO (c)
    raise NotImplementedError


def augmented_pd(q, qd, q_ref, qd_ref, qdd_ref, kp, kd, dynamics):
    """Add PD torque to the dynamics feedforward term.

    Args:
        q, qd: Measured NumPy arrays (6,) in radians and rad/s.
        q_ref, qd_ref, qdd_ref: Desired NumPy arrays (6,) in radians,
            rad/s, and rad/s^2.
        kp, kd: NumPy arrays (6,) of per-joint gains in N m/rad and
            N m s/rad.
        dynamics: Nominal model callable function as described in forward_dynamics.
            Can be called as dynamics(q, qd)

    Returns:
        NumPy array (6,) of joint torques in N m.
    """
    # TODO (e)
    raise NotImplementedError
