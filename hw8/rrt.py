"""HW8: collision checking, RRT, and trajectory refinement.

Run python hw8/rrt.py --demo 2d, 3d, or trajopt from the semester root.
All plotting, robot geometry, and optimizer setup are provided.
"""

import argparse
from pathlib import Path
from secrets import randbits

import numpy as np

from planning import robot_parameters as robot
from planning.robot_parameters import (
    COARSE_RADII, FINE_RADII, GOAL_SAMPLE_PROBABILITY, JOINT_BOUNDS,
    LINK_SPHERES, SELF_COLLISION_PAIRS, robot_sphere_centers,
    prepare_trajopt, resample_path, solve_trajopt,
)



# ============================= YOUR CODE =============================


def rrt(start, goal, edge_in_collision, bounds, step_size=0.15, max_iterations=2000, seed=0):
    """Grow an RRT using a provided edge-collision function.

    Args:
        start, goal: Finite NumPy arrays (D,). Use D=2 for the point robot
            or D=6 for UR5 joint configurations.
        edge_in_collision: edge_in_collision(start, end) returning True if a segment
            collides.
        bounds: Array (D, 2); row i contains the lower and upper coordinate limits.
        step_size: Positive largest length of our tree expansion.
        max_iterations: Nonnegative integer number of sampling attempts.
        seed: Integer seed for np.random.default_rng(seed).

    Returns:
        A dictionary: {"path": path, "nodes": nodes, "parents": parents}.
        nodes is (N, D), begins with start, and contains accepted nodes only.
        parents is an integer array (N,), with parents[0] == -1; every other
        parent precedes its child. path is (M, D), starts at start, and ends
        exactly at goal, or is None if no solution is found. For example,
        an unobstructed solution may have path = np.array([[0., 0.], [1., 0.]]).

        Out-of-bounds or colliding endpoints return path=None with only the
        root node. Sample goal with probability GOAL_SAMPLE_PROBABILITY, or
        sample uniformly in bounds. Endpoint checks, direct connections, goal
        connection, and path reconstruction are provided. Do not modify inputs.
    """
    start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    bounds = np.asarray(bounds)
    nodes, parents, initial_result = _initialize_rrt(start, goal, edge_in_collision, bounds)
    if initial_result is not None:
        return initial_result
    rng = np.random.default_rng(seed)
    for _ in range(max_iterations):
        # TODO: sample a target, find its nearest node, and steer by at most step_size.
        # Skip zero-length or colliding extensions. Append candidate and its parent.
        raise NotImplementedError # Remove this line when function is completed

        # Provided: Goal connection and path reconstruction
        if np.linalg.norm(candidate - goal) <= step_size and not edge_in_collision(candidate, goal):
            if np.array_equal(candidate, goal):
                return _rrt_result(nodes, parents, len(nodes) - 1)
            nodes.append(goal.copy())
            parents.append(len(nodes) - 2)
            return _rrt_result(nodes, parents, len(nodes) - 1)
    return _rrt_result(nodes, parents)


def robot_in_collision(theta, obstacles):
    """Check a UR5 configuration using the provided sphere hierarchy.

    Args:
        theta: NumPy array (6,), joint angles in radians.
        obstacles: NumPy array (K, 4), rows [x, y, z, radius] in meters in the
            robot base frame.

    Returns:
        Boolean True for an out-of-bounds configuration, robot-obstacle overlap,
        or a self-collision in SELF_COLLISION_PAIRS. Tangency is collision.


    fine and coarse center arrays returned by robot_sphere_centers(theta).
    LINK_SPHERES[i] picks the fine spheres corresponding to a coarse sphere i
    FINE_RADII and COARSE_RADII hold their radii. SELF_COLLISION_PAIRS lists
    link-index pairs that must be checked
    spheres_overlap(c1, r1, c2, r2) below implements the primitive test.
    """
    if np.any(theta < JOINT_BOUNDS[:, 0]) or np.any(theta > JOINT_BOUNDS[:, 1]):
        return True
    centers = robot_sphere_centers(theta)
    fine, coarse = centers["fine"], centers["coarse"]
    for link, indices in enumerate(LINK_SPHERES):
        for obstacle in obstacles:
            # TODO: skip this obstacle if the coarse sphere does not overlap it.
            raise NotImplementedError
            for index in indices:
                # TODO: return True if this fine sphere overlaps the obstacle.
                raise NotImplementedError
    for first, second in SELF_COLLISION_PAIRS:
        # TODO: skip this pair if its coarse spheres do not overlap.
        raise NotImplementedError
        for a in LINK_SPHERES[first]:
            for b in LINK_SPHERES[second]:
                # TODO: return True if these fine spheres overlap.
                raise NotImplementedError
    return False


def interpolate_joint_path(theta_start, theta_end, max_joint_step):
    """Sample a straight segment in joint space for collision checking.

    Args:
        theta_start, theta_end: Finite NumPy arrays with shape (6,), in radians.
        max_joint_step: Positive scalar maximum change in any one joint between
            consecutive samples. This is not a timestep or a playback speed.

    Returns:
        NumPy array (N, 6) including both endpoints in order, using equally
        spaced linear interpolation. Use the smallest number of intervals
        satisfying max(abs(theta[i+1] - theta[i])) <= max_joint_step.
        Identical endpoints produce one row. For example, changing only the
        first joint from 0 to 0.2 with max_joint_step=0.1 gives three rows.
        Do not wrap angles; the planner uses bounded joint intervals.
        np.ceil and np.linspace may be useful.
    """
    if max_joint_step <= 0:
        raise ValueError("max_joint_step must be positive")
    theta_start = np.asarray(theta_start, dtype=float)
    theta_end = np.asarray(theta_end, dtype=float)
    if np.array_equal(theta_start, theta_end):
        return np.array([theta_start])
    # TODO: calculate the interval count and return equally spaced samples.
    raise NotImplementedError


def smooth_path(path):
    """Refine a collision-free UR5 RRT path with the provided TrajOpt backend.

    Args:
        path: Finite NumPy array (N, 6), N >= 1, in JOINT_NAMES order. It must
            respect JOINT_BOUNDS and be collision-free in robot.OBSTACLES.

    Returns:
        The candidate joint path (M, 6) returned by solve_trajopt, or None if
        the solver fails.

    Call resample_path(path, count=40), prepare_trajopt(initial_trajectory), and then
    solve_trajopt(problem). Return the solver result. The provided request
    fixes the endpoints and handles constant paths without optimization.
    """
    # TODO
    return ...


# ============================= PROVIDED =============================

def _rrt_result(nodes, parents, index=None):
    """Build the RRT result from node and parent lists.

    nodes contains arrays (D,), parents contains integer indices with root -1,
    and index is the goal node or None for failure. Returns the documented
    dictionary with path (M, D) or None, nodes (N, D), and parents (N,).
    """
    path = None
    if index is not None:
        path = []
        while index != -1:
            path.append(nodes[index].copy())
            index = parents[index]
        path = np.array(path[::-1])
    return {"path": path, "nodes": np.array(nodes), "parents": np.array(parents, dtype=int)}


def _initialize_rrt(start, goal, edge_in_collision, bounds):
    """Initialize an RRT and handle endpoints and direct connections.

    start/goal are arrays (D,), bounds is (D, 2), and edge_in_collision tests
    a segment. Returns (nodes, parents, initial_result). initial_result is a
    finished result dictionary, or None when the sampling loop is needed.
    """
    nodes, parents = [start.copy()], [-1]
    if (np.any(start < bounds[:, 0]) or np.any(start > bounds[:, 1])
            or np.any(goal < bounds[:, 0]) or np.any(goal > bounds[:, 1])
            or edge_in_collision(start, start) or edge_in_collision(goal, goal)):
        return nodes, parents, _rrt_result(nodes, parents)
    if np.array_equal(start, goal):
        return nodes, parents, _rrt_result(nodes, parents, 0)
    if not edge_in_collision(start, goal):
        nodes.append(goal.copy())
        parents.append(0)
        return nodes, parents, _rrt_result(nodes, parents, 1)
    return nodes, parents, None


def collision_detection_2d(start, end, center, radius):
    """Check whether a finite line segment touches or intersects a circle.

    Args:
        start, end: NumPy arrays with shape (2,), the segment endpoints.
        center: NumPy array with shape (2,), the circle center.
        radius: Nonnegative circle radius, in the same units as the points.

    Returns:
        Boolean: True for collision, including tangency and endpoints
        inside the circle. Identical endpoints check a single point.
    """
    direction = end - start
    length_squared = direction @ direction
    if length_squared == 0:
        closest = start
    else:
        fraction = np.clip((center - start) @ direction / length_squared, 0., 1.)
        closest = start + fraction * direction
    delta = closest - center
    return bool(delta @ delta <= radius ** 2)

def spheres_overlap(first, first_radius, second, second_radius):
    """Return whether two spheres touch: centers (3,), nonnegative scalar radii.

    Coordinates and radii use meters. Returns a Boolean, including tangency.
    """
    delta = first - second
    return bool(delta @ delta <= (first_radius + second_radius) ** 2)


def circles_edge_in_collision(start, end, circles, collision=collision_detection_2d):
    """Check a 2D segment against circles (K, 3), rows [x, y, radius].

    start and end are arrays (2,). collision is the provided segment-circle
    function. Returns True if any circle collides, False for an empty scene.
    """
    return any(collision(start, end, circle[:2], circle[2]) for circle in circles)


def scene_2d(name):
    """Return start (2,), goal (2,), bounds (2, 2), and circles (K, 3).

    name is "open", "blocked", or "passage". All coordinates are in meters.
    """
    scenes = {
        "open": np.empty((0, 3)),
        "blocked": np.array([[0.5, 0.5, 0.22]]),
        "passage": np.array([[0.45, 0.25, 0.18], [0.55, 0.75, 0.18]]),
    }
    return np.array([0.08, 0.1]), np.array([0.92, 0.9]), np.array([[0., 1.], [0., 1.]]), scenes[name]


def main():
    """Run a provided 2D, 3D, or TrajOpt demo; nothing runs at import time."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", choices=("2d", "3d", "collision", "interpolation", "trajopt"),
                        default="2d")
    parser.add_argument("--scene", choices=("open", "blocked", "passage"), default="blocked")
    parser.add_argument("--seed", type=int,
                        help="RRT seed (default: random for TrajOpt, 0 for other demos)")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--step-size", type=float)
    parser.add_argument("--reference", action="store_true", help="Staff repository only")
    parser.add_argument("--no-visualize", action="store_true")
    parser.add_argument("--target", type=Path, help="Optional desired base-frame tool pose (4, 4), .npy")
    args = parser.parse_args()
    if args.seed is None:
        args.seed = randbits(32) if args.demo == "trajopt" else 0
    if args.demo == "trajopt":
        print(f"RRT random seed: {args.seed}")
    import sys
    implementation = sys.modules[__name__]
    if args.reference:
        import importlib.util
        location = Path(__file__).resolve().parents[2] / "autograders/hw8/source/reference.py"
        if not location.is_file():
            parser.error("--reference is available only in the staff repository")
        spec = importlib.util.spec_from_file_location("hw8_reference", location)
        implementation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(implementation)
    from planning import visualize
    visualize.run_demo(implementation, args)


if __name__ == "__main__":
    main()
