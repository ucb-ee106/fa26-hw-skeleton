"""All scored HW8 cases; numerical answers are precomputed, not implementations.

Optimizer checks use controlled public fixtures, not solver convergence.
"""

from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np


FUNCTION_NAMES = ("rrt", "robot_in_collision", "interpolate_joint_path", "smooth_path")
CASES = json.loads((Path(__file__).resolve().parent / "hw8_cases.json").read_text())
ATOL = 1e-7


def fixture_edge(mode):
    """Return a public test callback: free, blocked, goal-blocked, or a wall.

    The wall is x=0, -0.4 <= y <= 0.4. Other coordinates are irrelevant.
    This fixture is not the student's segment-circle collision implementation.
    """
    def edge(start, end):
        if mode == "free":
            return False
        if mode == "blocked":
            return True
        if mode == "goal_blocked":
            return bool(start[0] > 0.7 or end[0] > 0.7)
        if start[0] == end[0]:
            return bool(start[0] == 0 and min(start[1], end[1]) <= 0.4 and max(start[1], end[1]) >= -0.4)
        fraction = -start[0] / (end[0] - start[0])
        return bool(0 <= fraction <= 1 and abs(start[1] + fraction * (end[1] - start[1])) <= 0.4)
    return edge


def rrt_output_matches(actual, arguments, expected_success):
    """Validate tree/path properties without requiring one particular search tree."""
    if not isinstance(actual, dict) or not {"path", "nodes", "parents"} <= actual.keys():
        return False
    start, goal, bounds = arguments["start"], arguments["goal"], arguments["bounds"]
    nodes, parents = np.asarray(actual["nodes"]), np.asarray(actual["parents"])
    edge = arguments["edge_in_collision"]
    if (nodes.ndim != 2 or nodes.shape[1:] != start.shape or not len(nodes)
            or parents.shape != (len(nodes),) or not np.issubdtype(parents.dtype, np.integer)
            or not np.isrealobj(nodes) or not np.isfinite(nodes).all()
            or not np.allclose(nodes[0], start, atol=ATOL, rtol=0) or parents[0] != -1):
        return False
    invalid = (np.any(start < bounds[:, 0]) or np.any(start > bounds[:, 1])
               or np.any(goal < bounds[:, 0]) or np.any(goal > bounds[:, 1])
               or edge(start, start) or edge(goal, goal))
    if invalid:
        return actual["path"] is None and len(nodes) == 1
    if (len(nodes) > arguments["max_iterations"] + 2 or np.any(nodes < bounds[:, 0] - ATOL)
            or np.any(nodes > bounds[:, 1] + ATOL)):
        return False
    direct = not edge(start, goal)
    for index in range(1, len(nodes)):
        parent = parents[index]
        if not 0 <= parent < index or edge(nodes[parent], nodes[index]):
            return False
        direct_goal = direct and parent == 0 and np.allclose(nodes[index], goal, atol=ATOL, rtol=0)
        if not direct_goal and np.linalg.norm(nodes[index] - nodes[parent]) > arguments["step_size"] + ATOL:
            return False
    path = actual["path"]
    if path is None:
        return expected_success is not True
    if expected_success is False:
        return False
    path = np.asarray(path)
    if (path.ndim != 2 or path.shape[1:] != start.shape or not len(path)
            or not np.isrealobj(path) or not np.isfinite(path).all()
            or not np.allclose(path[0], start, atol=ATOL, rtol=0)
            or not np.allclose(path[-1], goal, atol=ATOL, rtol=0)):
        return False
    for end, node in enumerate(nodes):
        if not np.allclose(node, goal, atol=ATOL, rtol=0):
            continue
        chain = []
        while end != -1:
            chain.append(nodes[end])
            end = parents[end]
        if path.shape == (len(chain), len(start)) and np.allclose(path, chain[::-1], atol=ATOL, rtol=0):
            return True
    return False


@contextmanager
def patched_values(module, values):
    """Install fixture geometry/helpers temporarily and restore every value."""
    saved = []
    try:
        for namespace in (module, module.robot, np.random):
            for name, value in values.items():
                if hasattr(namespace, name):
                    saved.append((namespace, name, getattr(namespace, name)))
                    setattr(namespace, name, value)
        yield
    finally:
        for namespace, name, value in reversed(saved):
            setattr(namespace, name, value)


def evaluate_case(module, name, case):
    """Run one public case and return (passed, actual output or exception)."""
    arguments = {key: np.asarray(value) if isinstance(value, list) else value
                 for key, value in case["inputs"].items()}
    if "obstacles" in arguments and arguments["obstacles"].size == 0:
        arguments["obstacles"] = arguments["obstacles"].reshape(0, 4)
    before = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in arguments.items()}
    expected, values, calls = case["expected"], {}, []
    if name == "rrt":
        arguments["edge_in_collision"] = fixture_edge(case["callback"])
        if case.get("repeat_start"):
            from types import SimpleNamespace
            generator = SimpleNamespace(random=lambda: .5,
                                        uniform=lambda _low, _high: before["start"].copy())
            values["default_rng"] = lambda _seed: generator
    if "geometry" in case:
        geometry = case["geometry"]
        for key in ("FINE_RADII", "COARSE_RADII", "JOINT_BOUNDS"):
            values[key] = np.asarray(geometry[key])
        values["LINK_SPHERES"] = [np.asarray(indices, dtype=int) for indices in geometry["LINK_SPHERES"]]
        values["SELF_COLLISION_PAIRS"] = [tuple(pair) for pair in geometry["SELF_COLLISION_PAIRS"]]
        centers = {key: np.asarray(value) for key, value in geometry["centers"].items()}
        values["robot_sphere_centers"] = lambda _theta: centers
    if name == "smooth_path":
        candidate = None if case.get("candidate") is None else np.array(case["candidate"], dtype=float)

        def prepare(seed):
            calls.append("prepare")
            if seed.shape != (40, 6) or not np.allclose(seed, module.resample_path(before["path"], 40)):
                raise AssertionError("Use the resampled input path as the optimizer seed")
            return {"seed": seed.copy()}

        def solve(problem):
            calls.append("solve")
            if "seed" not in problem:
                raise AssertionError("Pass the request returned by prepare_trajopt")
            return None if candidate is None else candidate.copy()

        values.update(prepare_trajopt=prepare, solve_trajopt=solve)
    try:
        with patched_values(module, values):
            actual = getattr(module, name)(**arguments)
        if name == "rrt":
            passed = rrt_output_matches(actual, dict(before, edge_in_collision=arguments["edge_in_collision"]), expected)
        elif name in ("collision_detection_2d", "robot_in_collision"):
            passed = isinstance(actual, (bool, np.bool_)) and bool(actual) == expected
        elif expected is None:
            passed = actual is None
        else:
            array, answer = np.asarray(actual), np.asarray(expected)
            passed = (array.shape == answer.shape and np.isrealobj(array) and np.isfinite(array).all()
                      and np.allclose(array, answer, atol=ATOL, rtol=ATOL))
        if name == "smooth_path":
            passed &= calls == ["prepare", "solve"]
        if name in ("rrt", "smooth_path"):
            passed &= all(np.array_equal(arguments[key], value) for key, value in before.items()
                          if isinstance(value, np.ndarray))
        return bool(passed), actual
    except BaseException as error:
        return False, f"{type(error).__name__}: {error}"


def run_checks(module=None):
    """Run all public cases and show one failed example per function."""
    if module is None:
        import rrt as module
    total = passed_count = 0
    for name in FUNCTION_NAMES:
        shown_failure = False
        for case in CASES[name]:
            passed, actual = evaluate_case(module, name, case)
            total += 1
            passed_count += passed
            print(f"[{'PASS' if passed else 'FAIL'}] {name}: {case['name']}")
            if not passed and not shown_failure:
                print("  Inputs:", case["inputs"])
                print("  Expected:", case["expected"])
                print("  Actual:", actual)
                shown_failure = True
    print(f"{passed_count}/{total} HW8 checks passed.")
    return passed_count == total


if __name__ == "__main__":
    raise SystemExit(0 if run_checks() else 1)
