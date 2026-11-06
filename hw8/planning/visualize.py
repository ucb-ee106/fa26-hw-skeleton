"""Provided 2D RRT plots and Swift UR5 playback for HW8."""

from time import monotonic

import numpy as np

from . import robot_parameters as robot


def visualize_2d(result, start, goal, bounds, circles):
    """Show an RRT result dictionary for a point robot using Matplotlib.

    start/goal are (2,), bounds is (2, 2), and circles is (K, 3).
    Displays accepted edges, obstacles, and a final path if one was found.
    Returns (figure, axes); the caller decides when to call plt.show().
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    figure, axes = plt.subplots()
    for circle in circles:
        axes.add_patch(Circle(circle[:2], circle[2], color="tab:red", alpha=0.5))
    for index, parent in enumerate(result["parents"]):
        if parent >= 0:
            edge = result["nodes"][[parent, index]]
            axes.plot(edge[:, 0], edge[:, 1], color="0.7", linewidth=0.7)
    axes.plot([], [], color="0.7", label="Accepted RRT edges")
    if result["path"] is not None:
        path = result["path"]
        axes.plot(path[:, 0], path[:, 1], color="tab:blue", linewidth=2.5, label="Final path")
    axes.scatter(*start, color="tab:green", marker="o", label="Start", zorder=5)
    axes.scatter(*goal, color="tab:purple", marker="*", s=100, label="Goal", zorder=5)
    axes.set(xlim=bounds[0], ylim=bounds[1], xlabel="x (m)", ylabel="y (m)",
             title="RRT: path found" if result["path"] is not None else "RRT: no path found")
    axes.set_aspect("equal")
    axes.legend()
    figure.tight_layout()
    return figure, axes


def visualize_robot(paths, browser=None, test=False, collision_checks=None):
    """Play named UR5 joint paths with shared progress and sphere overlays.

    Args:
        paths: Dictionary of names to finite arrays (N, 6), N >= 1. All paths
            use the same scene. They are compared by joint-space arc-length
            progress, not physical time or estimated execution speed.
        browser: Optional Swift browser name.
        test: If True, initialize headlessly and return handles without looping.
        collision_checks: Optional dictionary of Boolean results for each
            configuration in paths. When supplied, play only those samples.

    Returns:
        A dictionary of environment, robot handle, and callbacks in test mode.
        Otherwise runs until the viewer is closed. Space is displayed in meters.
    """
    import roboticstoolbox as rtb
    import spatialgeometry as sg
    import swift
    from spatialmath import SE3

    model = robot.load_visual_robot()
    environment = swift.Swift()
    environment.launch(headless=test, browser=browser, axes=False, ground_pattern="@grid")
    environment.set_camera_pose([1.65, -1.5, 1.1], [0.25, 0.0, 0.35])
    names = list(paths)
    options = names + (["Compare"] if len(names) > 1 and collision_checks is None else [])
    sampled = paths if collision_checks is not None else {
        name: robot.resample_path(path, 300) for name, path in paths.items()
    }
    state = {"playing": not test, "progress": 0.0, "started": monotonic(), "selection": names[0]}
    fine = [sg.Sphere(float(radius), color=[0.1, 0.6, 0.9, 0.3]) for radius in robot.FINE_RADII]
    coarse = [sg.Sphere(float(radius), color=[0.95, 0.65, 0.1, 0.2]) for radius in robot.COARSE_RADII]
    state["overlay_shapes"] = []
    for index, obstacle in enumerate(robot.OBSTACLES):
        environment.add_shape(sg.Sphere(float(obstacle[3]), pose=SE3(*obstacle[:3]),
                                       color=[0.85, 0.18, 0.18, 0.95]), name=f"Obstacle {index + 1}")
    traces, markers = {}, {}
    colors = ([0.1, 0.4, 0.95], [0.1, 0.75, 0.3])
    for index, name in enumerate(names if collision_checks is None else []):
        points = np.array([robot.link_poses(q)["ee_link"][:3, 3] for q in sampled[name]])
        trace = sg.Polyline(points.T, linewidth=3, color=colors[index % 2])
        environment.add_shape(trace, name=name)
        traces[name] = trace
        marker = sg.Sphere(.018, color=colors[index % 2] + [1.0])
        environment.add_shape(marker, name=f"{name} tool position")
        markers[name] = marker
    legend = "; ".join(f"{color}: {name}" for color, name in zip(("Blue", "green"), names))
    label = swift.Label(label=legend, compact=True)
    environment.add_ui(label)

    def update(progress):
        state["progress"] = float(np.clip(progress, 0, 1))
        selected = names[0] if state["selection"] == "Compare" else state["selection"]
        count = len(sampled[selected])
        if collision_checks is None:
            frame = int(state["progress"] * (count - 1))
        else:
            frame = min(int(state["progress"] * count), count - 1)
        theta = sampled[selected][frame]
        for name, marker in markers.items():
            marker.T = SE3(*robot.link_poses(sampled[name][frame])["ee_link"][:3, 3])
        centers = robot.robot_sphere_centers(theta)
        for shape, center in zip(fine, centers["fine"]):
            shape.T = SE3(*center)
        for shape, center in zip(coarse, centers["coarse"]):
            shape.T = SE3(*center)
        if collision_checks is not None:
            checks = collision_checks[selected]
            status = "Collision" if checks[frame] else "Clear"
            label.label = (f"<b>{selected}</b><br>Sample {frame + 1} of {count}: {status}"
                           f"<br>{sum(checks)} of {count} samples in collision")
        else:
            detail = "dots show both tool positions" if len(names) > 1 else "dot shows the tool position"
            label.label = (f"<b>{state['selection']}</b> — {100 * state['progress']:.0f}%<br>"
                           f"{legend}<br>Arm follows {selected}; {detail}.")
        return theta

    def animate(_time, _values):
        progress = state["progress"]
        if state["playing"]:
            progress = min((monotonic() - state["started"]) / 8.0, 1.0)
            if progress == 1:
                state["playing"] = False
                play.label = "Play"
        return update(progress)

    model.q = sampled[names[0]][0]
    handle = environment.add_robot(model, readonly=True, callback=animate, name="UR5")

    def restart(_value):
        state["started"] = monotonic()
        state["playing"] = True
        handle.q = update(0)
        play.label = "Pause"

    def toggle(_value):
        if state["progress"] >= 1:
            restart(None)
            return
        state["playing"] = not state["playing"]
        state["started"] = monotonic() - 8 * state["progress"]
        play.label = "Pause" if state["playing"] else "Play"

    def select(index):
        state["selection"] = options[int(index)]
        handle.q = update(state["progress"])

    def overlay(index):
        for shape in state["overlay_shapes"]:
            environment.remove(shape)
        state["overlay_shapes"] = fine if int(index) == 1 else coarse if int(index) == 2 else []
        for shape in state["overlay_shapes"]:
            environment.add_shape(shape)

    play = swift.Button(toggle, label="Pause" if not test else "Play")
    for control in (swift.Select(select, options=options,
                                 label="Path" if collision_checks is None else "Samples"), play,
                    swift.Button(restart, label="Restart"),
                    swift.Select(overlay, options=["Mesh only", "Fine spheres", "Coarse bounds"], label="Collision geometry")):
        environment.add_ui(control)
    handle.q = update(0)
    if test:
        environment.step(0)
        return {"environment": environment, "handle": handle, "state": state,
                "restart": restart, "select": select, "overlay": overlay, "update": update}
    try:
        environment.run(dt=0.03)
    finally:
        environment.close()


def run_demo(implementation, args):
    """Run the selected CLI demo using functions from implementation.

    args is the argparse namespace from rrt.main. Raises a clear error if a
    required student function is unfinished; no optimizer loads for 2D/3D RRT.
    """
    import rrt

    if args.demo in ("collision", "interpolation"):
        if args.demo == "collision":
            paths = {"Collision sweep": np.linspace(robot.START, robot.GOAL, 31)}
        else:
            paths = {
                f"Maximum joint step: {step:g} rad": implementation.interpolate_joint_path(
                    robot.START, robot.GOAL, step
                )
                for step in (0.5, robot.MAX_JOINT_STEP)
            }
        checks = {
            name: [implementation.robot_in_collision(theta, robot.OBSTACLES) for theta in path]
            for name, path in paths.items()
        }
        for name, results in checks.items():
            print(f"{name}: {sum(results)} of {len(results)} samples in collision")
        if not args.no_visualize:
            visualize_robot(paths, collision_checks=checks)
        return

    if args.demo == "2d":
        start, goal, bounds, circles = rrt.scene_2d(args.scene)
        edge = lambda a, b: rrt.circles_edge_in_collision(a, b, circles, implementation.collision_detection_2d)
        result = implementation.rrt(start, goal, edge, bounds,
                                      args.step_size if args.step_size is not None else 0.15,
                                      args.iterations if args.iterations is not None else 2000, args.seed)
        if not isinstance(result, dict):
            raise RuntimeError("Complete rrt before running this demo")
        print(f"2D RRT: {len(result['nodes'])} nodes; path found={result['path'] is not None}")
        if not args.no_visualize:
            import matplotlib.pyplot as plt
            visualize_2d(result, start, goal, bounds, circles)
            plt.show()
        return
    goal = robot.GOAL.copy()
    if args.target is not None:
        import roboticstoolbox as rtb
        from spatialmath import SE3
        target = np.load(args.target)
        if target.shape != (4, 4):
            raise ValueError("Target must be a base-frame (4, 4) pose")
        model = robot.load_visual_robot()
        solution = model.ikine_LM(SE3(target), q0=goal, end="ee_link")
        if not solution.success:
            raise RuntimeError("Provided IK could not find a goal configuration")
        goal = solution.q
    edge = lambda a, b: robot.robot_edge_in_collision(a, b, implementation.interpolate_joint_path,
                                                     implementation.robot_in_collision)
    bounds = robot.JOINT_BOUNDS if args.target is not None else robot.PLANNING_BOUNDS
    result = implementation.rrt(robot.START, goal, edge, bounds,
                                  args.step_size if args.step_size is not None else 0.35,
                                  args.iterations if args.iterations is not None else 4000, args.seed)
    if not isinstance(result, dict):
        raise RuntimeError("Complete rrt before running this demo")
    path = result["path"]
    print(f"3D RRT: {len(result['nodes'])} nodes; path found={path is not None}")
    if path is None:
        return
    if not robot.path_is_valid(path, implementation.interpolate_joint_path, implementation.robot_in_collision):
        raise RuntimeError("Returned RRT path failed independent dense collision checking")
    paths = {"RRT": path}
    if args.demo == "trajopt":
        refined = implementation.smooth_path(path)
        if refined is Ellipsis:
            raise RuntimeError("Complete smooth_path before running this demo")
        refined = robot.validate_refined_path(path, refined, implementation.interpolate_joint_path,
                                             implementation.robot_in_collision)
        if refined is None:
            print("TrajOpt failed validation; displaying the RRT path only")
        else:
            paths["TrajOpt"] = refined
            for name, values in paths.items():
                values = robot.resample_path(values)
                print(f"{name}: joint-space roughness={np.sum(np.diff(values, n=2, axis=0) ** 2):.6f}; dense check passed")
    if not args.no_visualize:
        visualize_robot(paths)
