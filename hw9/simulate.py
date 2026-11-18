"""Provided HW9 mass-spring plant, simulation, and plotting metrics.

The coordinate x measures displacement upslope. The plant has viscous damping:
m * x'' = -k * x - b * x' - m * g * sin(theta) + u.
"""

import numpy as np


M, K, B, THETA, G = 1.0, 2.0, 0.5, np.pi / 6, 10.0
D = -G * np.sin(THETA)
X_EQ = M * D / K
DT = 0.01
SUBSTEPS = 10


def plant_derivative(z, u):
    """Return state derivative (2,) for state z = [x, xdot] and force u.

    z is a NumPy array (2,) in meters and meters per second. u is a scalar
    force in newtons. The output contains velocity and acceleration.
    """
    x, xdot = z
    return np.array([xdot, (-K * x - B * xdot + u) / M + D])


def step_plant(z, u, dt):
    """Advance state z (2,) under constant scalar force u using RK4.

    dt is a positive interval in seconds. Returns a new NumPy array (2,).
    Each interval uses SUBSTEPS integration steps and leaves z unchanged.
    """
    h = dt / SUBSTEPS
    for _ in range(SUBSTEPS):
        k1 = plant_derivative(z, u)
        k2 = plant_derivative(z + 0.5 * h * k1, u)
        k3 = plant_derivative(z + 0.5 * h * k2, u)
        k4 = plant_derivative(z + h * k3, u)
        z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return z


def simulate(controller, x_ref=0.0, x0=X_EQ, T=6.0, noise_std=0.0,
             u_max=np.inf, seed=0, true_velocity=False):
    """Simulate the plant with a controller sampled every DT seconds.

    Args:
        controller: Object with update(x_meas, x_ref, xdot=None) returning force.
        x_ref: Scalar desired position in meters, or callable t -> position.
        x0: Scalar initial position in meters. Initial velocity is zero.
        T: Positive duration in seconds, with round(T / DT) >= 2.
        noise_std: Nonnegative position-noise standard deviation in meters.
        u_max: Nonnegative physical actuator limit in newtons, or np.inf.
            Use the same limit in the controller for back-calculation.
        seed: Integer seed for independent Gaussian position-noise samples.
        true_velocity: If True, pass exact velocity to controller.update.

    Returns:
        Dictionary of arrays (N,), where N = round(T / DT). Keys are t,
        x, xdot, y, u, u_cmd, and x_ref. Each row describes the start of
        a controller interval. y is the noisy position, u_cmd is controller
        output, and u is the force after physical actuator saturation.
    """
    rng = np.random.default_rng(seed)
    ref = x_ref if callable(x_ref) else lambda t: x_ref
    n = int(round(T / DT))
    t = np.arange(n) * DT
    x, xdot, y, u, u_cmd = (np.zeros(n) for _ in range(5))
    z = np.array([x0, 0.0])
    for i in range(n):
        x[i], xdot[i] = z
        y[i] = x[i] + noise_std * rng.standard_normal()
        if true_velocity:
            u_cmd[i] = controller.update(y[i], ref(t[i]), xdot[i])
        else:
            u_cmd[i] = controller.update(y[i], ref(t[i]))
        u[i] = np.clip(u_cmd[i], -u_max, u_max)
        z = step_plant(z, u[i], DT)
    return dict(t=t, x=x, xdot=xdot, y=y, u=u, u_cmd=u_cmd,
                x_ref=np.array([ref(ti) for ti in t]))


def overshoot(res):
    """Return scalar fractional overshoot for a constant-reference step.

    res is a simulation result dictionary. Normalize by the absolute change
    from initial position to setpoint. A zero-size step returns zero.
    """
    x, r0, r1 = res["x"], res["x"][0], res["x_ref"][-1]
    step = r1 - r0
    if abs(step) < 1e-9:
        return 0.0
    return float(max(0.0, np.max((x - r1) * np.sign(step)) / abs(step)))


def settling_time(res, tol=0.02):
    """Return sampled settling time in seconds for a constant-reference step.

    res is a simulation result dictionary and tol is a positive fractional
    tolerance. Use tol * abs(step) as the band, or tol meters for a zero step.
    Return np.inf if the final sample is outside the band, since settling
    was not observed. Return zero if all samples are inside the band.
    """
    step = abs(res["x_ref"][-1] - res["x"][0])
    band = tol * (step if step > 1e-9 else 1.0)
    outside = np.flatnonzero(np.abs(res["x"] - res["x_ref"][-1]) > band)
    if not len(outside):
        return 0.0
    if outside[-1] == len(res["t"]) - 1:
        return np.inf
    return float(res["t"][outside[-1] + 1])


def control_roughness(res):
    """Return RMS successive changes in controller output, in newtons.

    res is a simulation result dictionary with at least two samples.
    """
    return float(np.sqrt(np.mean(np.diff(res["u_cmd"]) ** 2)))
