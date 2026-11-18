"""HW9: implementing PID for the mass-spring system.

Complete finite_difference, LowPassFilter, and PIDController.
Run python hw9/run_pid.py from the semester repository root.
"""

import numpy as np


def finite_difference(x, x_prev, dt):
    """Estimate velocity from two position samples using a backward difference.

    Args:
        x, x_prev: Scalar current and previous positions, in meters.
        dt: Positive scalar sample period, in seconds.

    Returns:
        Scalar velocity in meters per second.
    """
    # TODO
    raise NotImplementedError


class LowPassFilter:
    """First-order low-pass filter for scalar velocity measurements.

    Discretize tau * v_f' + v_f = v using backward Euler.
    """

    def __init__(self, cutoff_hz, dt):
        """Initialize the filter with no previous measurement.

        Args:
            cutoff_hz: Positive scalar cutoff frequency, in Hz.
            dt: Positive scalar sample period, in seconds.

        Returns:
            None. Store the coefficient self.alpha and keep
            self.v_f as None until the first update.
        """
        self.v_f = None

        # TODO: compute and store alpha.
        self.alpha = None

    def update(self, v):
        """Filter one scalar velocity sample.

        Args:
            v: Scalar velocity measurement, in meters per second.

        Returns:
            Scalar filtered velocity. On the first call, set self.v_f to v.
        """
        # TODO
        raise NotImplementedError


class PIDController:
    """Discrete PID controller with derivative on the measured position.

    Each update computes e = x_ref - x_meas and a velocity estimate v.
    Use u_unsat = kp * e + I - kd * v, then clip to [-u_max, u_max].
    Finally update I += dt * (ki * e + kt * (u - u_unsat)).
    I already includes the integral gain. Preserve this order.
    """

    def __init__(self, kp, ki, kd, dt, u_max=np.inf, use_filter=False,
                 cutoff_hz=5.0, anti_windup=False, T_t=None):
        """Initialize a controller with zero integral and no previous position.

        Args:
            kp, ki, kd: Scalar proportional, integral, and derivative gains.
            dt: Positive scalar sample period, in seconds.
            u_max: Nonnegative scalar force limit, in newtons, or np.inf.
            use_filter: If True, filter finite-difference velocity estimates.
            cutoff_hz: Positive scalar filter cutoff frequency, in Hz.
            anti_windup: If True, enable back-calculation with kt = 1 / T_t.
            T_t: Positive scalar tracking time, in seconds, or None. For None,
                use sqrt(T_i * T_d), where T_i = kp / ki and T_d = kd / kp.
                This default requires positive kp, ki, and kd.

        Returns:
            None. Initialize self.kt, self.I, self.x_prev, and self.filter.
            With anti_windup=False, self.kt is zero and T_t is unused.
        """
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt, self.u_max = dt, u_max
        self.filter = LowPassFilter(cutoff_hz, dt) if use_filter else None
        # TODO (d): set kt for anti-windup, using the default T_t if needed.
        self.kt = 0.0
        self.I = 0.0
        self.x_prev = None

    def update(self, x_meas, x_ref, xdot=None):
        """Compute and return one control force, then save controller state.

        Args:
            x_meas: Scalar measured position, in meters.
            x_ref: Scalar desired position, in meters.
            xdot: Optional scalar exact velocity, in meters per second.
                The provided baseline uses this value directly and bypasses
                both finite differences and filtering.

        Returns:
            Scalar force u, in newtons, clipped to [-self.u_max, self.u_max].
            First compute the error and velocity. Without xdot, use zero
            velocity on the first call and finite_difference thereafter.
            Pass this estimate through self.filter when it is present.
            Save x_meas in self.x_prev on every call. Compute and clip the
            force using the current I, then update I using the saturated
            minus unsaturated force. Return the clipped force.

        Part (a) implements the update without filtering or anti-windup.
        Part (b) adds filtering. Part (d) sets kt in the constructor.

        Hint: Use np.clip(value, lower_bound, upper_bound) to limit the force.
        """
        # TODO
        raise NotImplementedError
