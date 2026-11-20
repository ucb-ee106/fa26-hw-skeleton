"""Run the HW10 shoulder pulse or UR5 tracking demo from the semester repo."""

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np
import sympy as sp

from simulate import (
    DT, HOLD_TIME, INERTIA_SCALES, PULSE_DURATION, PULSE_INITIAL, PULSE_START,
    TRANSFER_START, TRANSFER_TIME,
    shoulder_pulse, simulate, transfer_reference,
)
from ur5_model import GRAVITY, tool_home_pose, ur5_parameters


def build_models(implementation):
    """Generate NumPy dynamics and tool position functions once using SymPy."""
    theta, dtheta = sp.symbols("q:6"), sp.symbols("v:6")
    twists, homes, inertias, _ = ur5_parameters()
    M = sp.Matrix(implementation.inertia_matrix(theta, twists, homes, inertias))
    C = implementation.coriolis_matrix(M, theta, dtheta)
    G = implementation.gravity_vector(theta, twists, homes, inertias, GRAVITY)
    evaluate = sp.lambdify((theta, dtheta), (M, C, G), "numpy", cse=True, docstring_limit=0)
    pose = implementation.forward_kinematics(twists, theta) @ tool_home_pose()
    position = sp.lambdify((theta,), pose[:3, 3], "numpy", cse=True, docstring_limit=0)

    def dynamics(q, qd):
        mass, coriolis, gravity = evaluate(q, qd)
        return mass, coriolis, gravity.reshape(6)

    def tool_position(q):
        """Return the base-frame position (3,) of tool0 at joint angles q (6,)."""
        return position(q).reshape(3)

    return dynamics, tool_position


def run_pulse(implementation, dynamics):
    """Return the shoulder-pulse rollout at the shared simulation timestep.

    implementation supplies forward_dynamics and integrate_step; dynamics
    is the nominal plant. Each named rollout is a tuple from simulate.
    """
    gravity = dynamics(PULSE_INITIAL, np.zeros(6))[2]

    def control(t, _q, _qd):
        return shoulder_pulse(t, gravity)

    torque_at_onset = shoulder_pulse(PULSE_START, gravity)
    acceleration = implementation.forward_dynamics(PULSE_INITIAL, np.zeros(6), torque_at_onset, dynamics)
    print(f"Initial pulse acceleration: shoulder {acceleration[1]:+.3f}, "
          f"elbow {acceleration[2]:+.3f} rad/s^2")
    return {"Shoulder pulse": simulate(
        implementation.integrate_step, dynamics, control, PULSE_INITIAL, DT, PULSE_DURATION
    )}


def run_transfer(implementation, nominal, controllers, scales=INERTIA_SCALES):
    """Compare controllers using scaled link inertias on the nominal plant.

    Scaling every generalized link inertia by the same factor scales M,
    C, and G by that factor, so the symbolic model only needs to be built once.
    """
    runs = {}
    for name in controllers:
        if name == "computed":
            law = implementation.computed_torque
            label = "Computed torque"
            kp, kd = np.full(6, 36.0), np.full(6, 12.0)
        else:
            law = implementation.augmented_pd
            label = "Augmented PD"
            inertia = np.diag(nominal(TRANSFER_START, np.zeros(6))[0])
            rates = np.array([10.0, 10.0, 10.0, 5.0, 5.0, 5.0])
            kp, kd = rates**2 * inertia, 2 * rates * inertia

        for scale in scales:
            def model(q, qd):
                return tuple(scale * term for term in nominal(q, qd))

            def control(t, q, qd):
                return law(q, qd, *transfer_reference(t), kp, kd, model)

            runs[f"{label}, {scale:g}× inertia"] = simulate(
                implementation.integrate_step, nominal, control, TRANSFER_START,
                DT, TRANSFER_TIME + HOLD_TIME
            )
    return runs


def save_plot(demo, runs, tool_position, path):
    """Save one figure and print run metrics for the pulse or transfer.

    runs maps labels to tuples from simulate. tool_position maps joint
    arrays (6,) to positions (3,) in meters. path is the output PNG path.
    Returns None.
    """
    import matplotlib.pyplot as plt

    if demo == "pulse":
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        for t, q, _, tau in runs.values():
            axes[0].plot(t, q[:, 1] - q[0, 1], label="Shoulder")
            axes[0].plot(t, q[:, 2] - q[0, 2], "--", label="Elbow")
        axes[1].step(t, tau[:, 1] - tau[0, 1], where="post")
        axes[0].set_ylabel("Change in joint angle (rad)")
        axes[1].set_ylabel("Added shoulder torque (N m)")
    else:
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        print("\nTransfer errors: mean and max during the move, final after the hold.")
        print("Joint errors are absolute. Final joint error is the largest across the six joints.")
        header = (f"{'Run':<34} {'Mean joint (deg)':>17} {'Max joint (deg)':>16} "
                  f"{'Final joint (deg)':>17} {'Mean EE (mm)':>13} {'Max EE (mm)':>12} "
                  f"{'Final EE (mm)':>13}")
        print(header)
        print("-" * len(header))
        for label, (t, q, _, _) in runs.items():
            desired = np.array([transfer_reference(time)[0] for time in t])
            joint_error = np.degrees(np.abs(q - desired))
            tool_error = 1000 * np.array([np.linalg.norm(tool_position(actual) - tool_position(ref))
                                          for actual, ref in zip(q, desired)])
            axes[0].plot(t, joint_error.max(axis=1), label=label)
            axes[1].plot(t, joint_error.mean(axis=1))
            axes[2].plot(t, tool_error)
            moving = t <= TRANSFER_TIME
            move_joint_error = joint_error[moving]
            move_tool_error = tool_error[moving]
            print(f"{label:<34} {move_joint_error.mean():17.3f} {move_joint_error.max():16.3f} "
                  f"{joint_error[-1].max():17.3f} {move_tool_error.mean():13.2f} "
                  f"{move_tool_error.max():12.2f} {tool_error[-1]:13.2f}")
        for ax in axes:
            ax.axvline(TRANSFER_TIME, color="0.5", linestyle=":")
        axes[0].set_ylabel("Max joint error (deg)")
        axes[1].set_ylabel("Mean joint error (deg)")
        axes[2].set_ylabel("Tool position error (mm)")
    axes[-1].set_xlabel("Time (s)")
    if demo == "transfer":
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.87))
    else:
        axes[0].legend(fontsize=8)
        fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    """Run the requested HW10 demo, save a plot, and optionally launch Swift."""
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", choices=("pulse", "transfer"), default="transfer")
    parser.add_argument("--controller", choices=("both", "computed", "pd"), default="both")
    parser.add_argument("--inertia-scale", type=float, nargs="+", default=INERTIA_SCALES,
                        help="Controller link inertia scales (default: 1 1.001 1.01 1.1 1.25)")
    parser.add_argument("--no-viewer", action="store_true", help="Save plots and metrics without Swift")
    parser.add_argument("--output-dir", type=Path, default=directory / "img")
    parser.add_argument("--reference", action="store_true", help="Use the staff reference")
    args = parser.parse_args()
    if not all(np.isfinite(scale) and scale > 0 for scale in args.inertia_scale):
        parser.error("--inertia-scale values must be finite and positive")
    sys.path.insert(0, str(directory.parent))
    if args.reference:
        path = directory.parents[2] / "autograders/hw10/source/reference.py"
        if not path.is_file():
            parser.error("--reference is available in the staff repository")
        spec = importlib.util.spec_from_file_location("hw10_reference", path)
        implementation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(implementation)
    else:
        import hw10 as implementation
    try:
        print("Building nominal UR5 dynamics...", flush=True)
        nominal, tool_position = build_models(implementation)
        if args.demo == "pulse":
            runs = run_pulse(implementation, nominal)
        else:
            controllers = ("computed", "pd") if args.controller == "both" else (args.controller,)
            runs = run_transfer(implementation, nominal, controllers, args.inertia_scale)
    except NotImplementedError:
        parser.exit(1, "Finish the required HW10 functions, then run this demo again.\n")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_plot(args.demo, runs, tool_position, args.output_dir / f"hw10_{args.demo}.png")
    if not args.no_viewer:
        from visualize import play_runs

        play_runs(args.demo, runs, tool_position)


if __name__ == "__main__":
    main()
