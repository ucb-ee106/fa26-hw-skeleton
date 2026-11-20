"""Provided HW10 trajectories, torque inputs, and rollout loop."""

import numpy as np


START = np.array([-1.0, -1.35, 1.65, -1.75, -1.4, 0.0])


TRANSFER_START = START.copy()
GOAL = np.array([1.0, -1.18, 1.45, -1.65, -1.4, 0.5])
SWEEP_AMPLITUDES = np.array([0.25, 0.38, 0.35, 0.47, 0.44, 0.55])
SWEEP_PHASES = np.array([0, np.pi / 4, np.pi / 4, -np.pi / 4, 0, np.pi / 2])
SWEEP_CYCLES = 2
TRANSFER_TIME = 4.8
HOLD_TIME = 5.0
DT = 0.002
PULSE_INITIAL = np.array([-1.0, -2.15, 1.85, -1.75, -1.4, 0.0])
PULSE_START = 0.36
PULSE_STOP = 0.72
PULSE_DURATION = 1.08
INERTIA_SCALES = (1.0, 1.001, 1.01, 1.1, 1.25)


def transfer_reference(t):
    """Return desired (q, qd, qdd) at scalar time t in seconds.

    All three outputs are arrays (6,) in radians, rad/s, and rad/s^2.
    Each joint adds two periods of a sinusoid to the move from
    TRANSFER_START to GOAL. The quintic time scaling gives zero velocity
    and acceleration at both endpoints.
    """
    if t <= 0:
        return TRANSFER_START.copy(), np.zeros(6), np.zeros(6)
    if t >= TRANSFER_TIME:
        return GOAL.copy(), np.zeros(6), np.zeros(6)

    s = t / TRANSFER_TIME
    progress = s**3 * (10 - 15 * s + 6 * s**2)
    progress_rate = 30 * s**2 * (1 - s)**2 / TRANSFER_TIME
    progress_accel = 60 * s * (1 - s) * (1 - 2 * s) / TRANSFER_TIME**2
    frequency = 2 * np.pi * SWEEP_CYCLES
    phase = frequency * progress + SWEEP_PHASES
    change = GOAL - TRANSFER_START

    q = TRANSFER_START + progress * change + SWEEP_AMPLITUDES * (
        np.sin(phase) - np.sin(SWEEP_PHASES)
    )
    slope = change + frequency * SWEEP_AMPLITUDES * np.cos(phase)
    curvature = -frequency**2 * SWEEP_AMPLITUDES * np.sin(phase)
    qd = progress_rate * slope
    qdd = progress_accel * slope + progress_rate**2 * curvature
    return q, qd, qdd


def shoulder_pulse(t, gravity_hold):
    """Return gravity_hold (6,) plus 4 N m at the shoulder during the pulse.

    t is the time in seconds. The gravity compensation is fixed at its
    initial value. Returns a new torque array (6,) in N m.
    """
    tau = gravity_hold.copy()
    if PULSE_START <= t < PULSE_STOP:
        tau[1] += 4.0
    return tau


def simulate(step, dynamics, control, q0, dt, duration):
    """Roll out a six-joint arm starting at q0 (6,) with zero velocity.

    step is integrate_step(q, qd, tau, dt, dynamics); dynamics evaluates
    the plant M, C, G. control(t, q, qd) returns a torque array (6,) in N m.
    dt and duration are in seconds and duration is an integer multiple
    of dt. Return time (N,) and position, velocity, and torque (N, 6).
    Each row gives the state and controller output at the corresponding
    time, including the final time. Positions are in radians.
    """
    times = np.arange(round(duration / dt) + 1) * dt
    positions = np.empty((len(times), 6))
    velocities = np.zeros_like(positions)
    torques = np.empty_like(positions)
    positions[0] = q0
    for k, t in enumerate(times):
        torques[k] = control(t, positions[k], velocities[k])
        if k + 1 < len(times):
            positions[k + 1], velocities[k + 1] = step(
                positions[k], velocities[k], torques[k], dt, dynamics
            )
            if not np.all(np.isfinite(positions[k + 1])) or not np.all(np.isfinite(velocities[k + 1])):
                raise FloatingPointError(f"Nonfinite joint state at t={times[k + 1]:.3f} s")
    return times, positions, velocities, torques
