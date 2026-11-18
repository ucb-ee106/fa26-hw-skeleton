"""Run the provided HW9 PID scenarios and save figures for the write-up."""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulate import DT, simulate, overshoot, settling_time, control_roughness


KP, KI, KD = 11.5, 12.0, 4.5
NOISE = 2e-3
U_MAX = 8.0
CUTOFFS = [1.0, 5.0, 20.0]


def summary(name, result):
    """Print overshoot, sampled settling time, and force roughness for result.

    name is a string and result is a dictionary returned by simulate.
    Returns None. Settling is measured only over the simulated interval.
    """
    time = settling_time(result)
    settled = f"{time:.2f} s" if np.isfinite(time) else "not settled"
    print(f"{name:40s} overshoot {100 * overshoot(result):5.1f}%   "
          f"settling {settled:>11s}   roughness {control_roughness(result):5.2f} N")


def plot_runs(runs, filename, title, show_cmd=False):
    """Save position and force plots for a list of simulation results.

    runs contains (label, result, line_style) tuples. filename is the output
    path and title is a string. show_cmd selects controller output rather
    than applied force. Returns None; closes the figure after saving it.
    """
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for label, result, style in runs:
        axes[0].plot(result["t"], result["x"], style, lw=1.6, label=label)
        force = result["u_cmd"] if show_cmd else result["u"]
        axes[1].plot(result["t"], force, style, lw=1.2, label=label)
    axes[0].plot(runs[0][1]["t"], runs[0][1]["x_ref"], "k:", label="reference")
    axes[0].set_ylabel("position x (m)")
    axes[1].set_ylabel("commanded force u (N)" if show_cmd else "force u (N)")
    axes[1].set_xlabel("time (s)")
    axes[0].set_title(title)
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)


def main():
    """Run parts (a)-(d) with student code, or the private --reference code.

    Read --output-dir from the command line, defaulting to img beside this
    script. Save three PNG figures and print metrics. Returns None.
    Unimplemented parts are skipped so students can work incrementally.
    """
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", action="store_true", help="use the private staff reference")
    parser.add_argument("--output-dir", type=Path, default=directory / "img")
    args = parser.parse_args()
    if args.reference:
        path = directory.parents[1] / "autograders/hw9/source/reference.py"
        if not path.is_file():
            parser.error("--reference is available only in the staff repository")
        spec = importlib.util.spec_from_file_location("pid_reference", path)
        implementation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(implementation)
    else:
        import pid as implementation
    controller = implementation.PIDController
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"])

    noisy = None
    try:
        baseline = simulate(controller(KP, KI, KD, DT), true_velocity=True)
        clean = simulate(controller(KP, KI, KD, DT))
        noisy = simulate(controller(KP, KI, KD, DT), noise_std=NOISE)
        summary("baseline, true velocity", baseline)
        summary("(a) finite difference, no noise", clean)
        summary("(a) finite difference, 2 mm noise", noisy)
        plot_runs([("finite difference, 2 mm noise", noisy, "-"),
                   ("true velocity", baseline, "-"),
                   ("finite difference, no noise", clean, "--")],
                  args.output_dir / "pid_a_finite_difference.png",
                  "(a) Replacing the velocity sensor with a finite difference", True)
    except NotImplementedError:
        print("[baseline and (a)] skipped: not implemented yet")

    try:
        if noisy is None:
            raise NotImplementedError
        runs = [("2 mm noise, unfiltered", noisy, "-")]
        for cutoff in CUTOFFS:
            result = simulate(controller(KP, KI, KD, DT, use_filter=True,
                                         cutoff_hz=cutoff), noise_std=NOISE)
            summary(f"(b) filtered, f_c = {cutoff:g} Hz", result)
            runs.append((f"filtered, $f_c$ = {cutoff:g} Hz", result, "-"))
        plot_runs(runs, args.output_dir / "pid_b_low_pass.png",
                  "(b) Low-pass filtering the velocity estimate", True)
    except NotImplementedError:
        print("[(b)] skipped: not implemented yet")

    try:
        runs = []
        for anti_windup in (False, True):
            label = "tracking anti-windup" if anti_windup else "no anti-windup"
            result = simulate(controller(KP, KI, KD, DT, use_filter=True,
                                         cutoff_hz=5.0, u_max=U_MAX,
                                         anti_windup=anti_windup),
                              x_ref=1.0, noise_std=NOISE, u_max=U_MAX, T=8.0)
            part = "d" if anti_windup else "c"
            summary(f"({part}) saturated, {label}", result)
            runs.append((label, result, "-"))
        plot_runs(runs, args.output_dir / "pid_c_anti_windup.png",
                  r"(c) and (d) Step to x = 1 m with $|u| \leq 8$ N")
    except NotImplementedError:
        print("[(c) and (d)] skipped: not implemented yet")


if __name__ == "__main__":
    main()
