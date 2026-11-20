"""Provided Swift replay of HW10 joint histories."""

from pathlib import Path
import sys
from time import monotonic

import numpy as np

from simulate import transfer_reference


def play_runs(demo, runs, tool_position, headless=False):
    """Replay one of the named (time, q, qd, tau) rollouts in Swift.

    demo is "pulse" or "transfer". tool_position maps q (6,) to the
    tool position (3,) in meters. The viewer uses physical simulation time.
    With headless=True, initialize, step once, and close for a smoke test.
    Returns None.
    """
    import spatialgeometry as sg
    from spatialmath import SE3
    import swift

    # The HW8 model is a sibling of hw10 in the semester repository.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    try:
        from hw8.planning.robot_parameters import load_visual_robot
    finally:
        sys.path.pop(0)

    robot = load_visual_robot()
    env = swift.Swift()
    env.launch(headless=headless, axes=False, ground_pattern="@grid")
    try:
        env.set_camera_pose([1.5, -1.55, 1.2], [0.2, 0, 0.4])
        names = list(runs)
        selected = 0
        elapsed = 0.0
        started = monotonic()
        playing = not headless
        playback_rate = 0.5 if demo == "pulse" else 1.0
        status = swift.Label(label="", compact=True)
        env.add_ui(status)
        blue = [0.15, 0.42, 0.9]
        orange = [0.95, 0.42, 0.1]
        actual_marker = sg.Sphere(radius=0.012, color=orange)
        env.add_shape(actual_marker, name="Actual tool position")

        def sampled_tool_path(joints):
            indices = np.linspace(0, len(joints) - 1, min(len(joints), 160), dtype=int)
            return np.array([tool_position(joints[index]) for index in indices]).T

        if demo == "transfer":
            reference_marker = sg.Sphere(radius=0.012, color=blue)
            env.add_shape(reference_marker, name="Reference tool position")
            reference_joints = [transfer_reference(t)[0] for t in runs[names[0]][0]]
            env.add_shape(sg.Polyline(sampled_tool_path(reference_joints), linewidth=4, color=blue),
                          name="Reference tool path")
        actual_points = [sampled_tool_path(q) for _, q, _, _ in runs.values()]
        actual_path = sg.Polyline(actual_points[0], linewidth=3, color=orange)
        env.add_shape(actual_path, name="Selected actual tool path")

        def frame(_time, _values):
            nonlocal elapsed, playing
            times, q, _, _ = runs[names[selected]]
            if playing:
                elapsed = min(playback_rate * (monotonic() - started), times[-1])
                if elapsed >= times[-1]:
                    playing = False
                    play.label = "Play"
            angles = np.array([np.interp(elapsed, times, column) for column in q.T])
            actual = tool_position(angles)
            actual_marker.T = SE3(*actual)
            if demo == "transfer":
                target = transfer_reference(elapsed)[0]
                desired = tool_position(target)
                reference_marker.T = SE3(*desired)
                error = 1000 * np.linalg.norm(actual - desired)
                joint_errors = np.degrees(np.abs(angles - target))
                caption = ("<br>Blue: desired · orange: selected run"
                           f"<br>Tool error: {error:.1f} mm · joint error: "
                           f"{joint_errors.max():.2f}° max, {joint_errors.mean():.2f}° mean")
            else:
                shoulder, elbow = np.degrees(angles[1:3] - q[0, 1:3])
                caption = f"<br>Shoulder: {shoulder:+.1f}° · elbow: {elbow:+.1f}° from start"
            status.label = f"{names[selected]} · {elapsed:.2f} s{caption}"
            return angles

        def restart(_value):
            nonlocal elapsed, started, playing
            elapsed, started, playing = 0.0, monotonic(), True
            play.label = "Pause"
            handle.q = frame(None, None)

        def toggle(_value):
            nonlocal started, playing
            if elapsed >= runs[names[selected]][0][-1]:
                restart(None)
                return
            playing = not playing
            started = monotonic() - elapsed / playback_rate
            play.label = "Pause" if playing else "Play"

        def select(index):
            nonlocal selected, actual_path
            selected = int(index)
            env.remove(actual_path)
            actual_path = sg.Polyline(actual_points[selected], linewidth=3, color=orange)
            env.add_shape(actual_path, name="Selected actual tool path")
            restart(None)

        play = swift.Button(toggle, label="Pause" if playing else "Play")
        if len(names) > 1:
            env.add_ui(swift.Select(select, options=names, label="Run"))
        env.add_ui(play)
        env.add_ui(swift.Button(restart, label="Restart"))
        robot.q = runs[names[0]][1][0]
        handle = env.add_robot(robot, readonly=True, callback=frame, name="UR5")
        handle.q = frame(None, None)
        if headless:
            env.step(0)
        else:
            env.run(dt=0.03)
    finally:
        env.close()
