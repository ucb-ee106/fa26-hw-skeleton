"""Provided UR5 geometry and planning utilities for HW8.

Collision geometry: KavrakiLab/VAMP, revision recorded in assets/provenance.json.
TrajOpt: https://tesseract-robotics.github.io/tesseract_nanobind/
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


ASSETS = Path(__file__).resolve().parent / "assets"
DESCRIPTION = ET.parse(ASSETS / "robot_spheres.urdf").getroot()
JOINTS = list(DESCRIPTION.findall("joint"))
JOINT_NAMES = [joint.get("name") for joint in JOINTS if joint.get("type") == "revolute"]
LINK_NAMES = [link.get("name") for link in DESCRIPTION.findall("link") if link.findall("collision")]
JOINT_BOUNDS = np.array([[-np.pi, np.pi]] * 6)
PLANNING_BOUNDS = np.array([[-1.6, 1.6], [-2.0, -.8], [.8, 2.3],
                            [-2.2, -1.0], [-1.8, -.6], [-.5, .5]])
GOAL_SAMPLE_PROBABILITY = 0.1
MAX_JOINT_STEP = 0.04
START = np.array([-1.0, -1.4, 1.7, -1.8, -1.4, 0.0])
GOAL = np.array([1.0, -1.1, 1.5, -1.5, -1.1, 0.0])
OBSTACLES = np.array([[0.340674, 0.01615, 0.41704, 0.09]])


def _origin(element):
    result = np.eye(4)
    if element is not None:
        result[:3, :3] = Rotation.from_euler("xyz", np.fromstring(element.get("rpy", "0 0 0"), sep=" ")).as_matrix()
        result[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
    return result


JOINT_ORIGINS = [_origin(joint.find("origin")) for joint in JOINTS]
FINE_LOCAL = []
FINE_RADII = []
SPHERE_LINKS = []
for link in DESCRIPTION.findall("link"):
    for collision in link.findall("collision"):
        FINE_LOCAL.append(_origin(collision.find("origin"))[:3, 3])
        FINE_RADII.append(float(collision.find("geometry/sphere").get("radius")))
        SPHERE_LINKS.append(LINK_NAMES.index(link.get("name")))
FINE_LOCAL = np.array(FINE_LOCAL)
FINE_RADII = np.array(FINE_RADII)
SPHERE_LINKS = np.array(SPHERE_LINKS)
LINK_SPHERES = [np.flatnonzero(SPHERE_LINKS == index) for index in range(len(LINK_NAMES))]
COARSE_LOCAL = np.array([FINE_LOCAL[indices].mean(axis=0) for indices in LINK_SPHERES])
COARSE_RADII = np.array([
    max(np.linalg.norm(FINE_LOCAL[indices] - center, axis=1) + FINE_RADII[indices])
    for indices, center in zip(LINK_SPHERES, COARSE_LOCAL)
])
semantic = ET.parse(ASSETS / "robot.srdf").getroot()
ALLOWED_LINK_PAIRS = {
    frozenset((pair.get("link1"), pair.get("link2")))
    for pair in semantic.findall("disable_collisions")
}
SELF_COLLISION_PAIRS = [
    (first, second) for first in range(len(LINK_NAMES)) for second in range(first + 1, len(LINK_NAMES))
    if frozenset((LINK_NAMES[first], LINK_NAMES[second])) not in ALLOWED_LINK_PAIRS
]


def link_poses(theta):
    """Return URDF link poses for joint angles theta (6,) in radians.

    Returns a dictionary of link names to arrays with shape (4, 4).
    URDF roll/pitch/yaw uses extrinsic xyz, unlike HW3's intrinsic XYZ labels.
    """
    poses = {"base_link": np.eye(4)}
    for joint, origin in zip(JOINTS, JOINT_ORIGINS):
        motion = np.eye(4)
        if joint.get("type") == "revolute":
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            angle = theta[JOINT_NAMES.index(joint.get("name"))]
            motion[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
        poses[joint.find("child").get("link")] = poses[joint.find("parent").get("link")] @ origin @ motion
    return poses


def robot_sphere_centers(theta):
    """Compute the provided collision-sphere positions for one configuration.

    Args:
        theta: NumPy array with shape (6,), containing joint angles in radians.

    Returns:
        Dictionary with "fine" (S, 3) and "coarse" (L, 3) center arrays in
        meters in the robot base frame. Radii are FINE_RADII and COARSE_RADII;
        LINK_SPHERES maps each coarse sphere to its fine spheres.
    """
    poses = link_poses(theta)
    fine = np.empty_like(FINE_LOCAL)
    coarse = np.empty_like(COARSE_LOCAL)
    for index, name in enumerate(LINK_NAMES):
        pose = poses[name]
        indices = LINK_SPHERES[index]
        fine[indices] = FINE_LOCAL[indices] @ pose[:3, :3].T + pose[:3, 3]
        coarse[index] = pose[:3, :3] @ COARSE_LOCAL[index] + pose[:3, 3]
    return {"fine": fine, "coarse": coarse}


def load_visual_robot():
    """Load the bundled UR5 mesh model, using exactly the collision model's frames.

    Returns a Robotics Toolbox Robot with six joints in JOINT_NAMES order.
    Mesh paths are made absolute in a temporary URDF; no model download is needed.
    """
    from tempfile import TemporaryDirectory
    import roboticstoolbox as rtb
    from roboticstoolbox.models.URDF.URDFRobot import URDF_read

    description = ET.parse(ASSETS / "robot_visual.urdf")
    for mesh in description.findall(".//mesh"):
        mesh.set("filename", str(ASSETS / mesh.get("filename")))
    with TemporaryDirectory(prefix="hw8-visual-") as directory:
        filename = Path(directory) / "robot.urdf"
        description.write(filename, encoding="utf-8", xml_declaration=True)
        links, name, _path = URDF_read(filename)
        return rtb.Robot(links, name=name)


def robot_edge_in_collision(start, end, interpolate, collision, obstacles=None, max_joint_step=MAX_JOINT_STEP):
    """Check an edge using the student's interpolation and collision functions.

    Args:
        start, end: Joint arrays with shape (6,), in radians.
        interpolate: interpolate_joint_path(start, end, max_joint_step).
        collision: robot_in_collision(theta, obstacles), returning a Boolean.
        obstacles: Array (K, 4) of sphere centers and radii in meters; None
            selects OBSTACLES. An empty (0, 4) array removes environment obstacles.
        max_joint_step: Positive maximum per-joint interpolation increment.

    Returns:
        True if a sampled configuration is out of bounds or in collision.
        Finite-resolution checking can miss collisions between samples.
    """
    obstacles = OBSTACLES if obstacles is None else obstacles
    for theta in interpolate(start, end, max_joint_step):
        if np.any(theta < JOINT_BOUNDS[:, 0]) or np.any(theta > JOINT_BOUNDS[:, 1]):
            return True
        if collision(theta, obstacles):
            return True
    return False


def resample_path(path, count=40):
    """Resample joint path (N, 6) to count >= 2 equally spaced arc-length samples.

    Returns an array (count, 6), preserving both endpoints. A constant path
    produces repeated copies. This is geometry, not physical time scaling.
    """
    path = np.asarray(path, dtype=float)
    lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    if lengths[-1] == 0:
        return np.repeat(path[:1], count, axis=0)
    lengths, indices = np.unique(lengths, return_index=True)
    progress = np.linspace(0, lengths[-1], count)
    return np.column_stack([np.interp(progress, lengths, path[indices, joint]) for joint in range(6)])


def path_is_valid(path, interpolate, collision, obstacles=None, max_joint_step=0.01):
    """Return whether every edge of nonempty joint path (N, 6) passes a dense check.

    interpolate and collision are the same callbacks as robot_edge_in_collision.
    obstacles=None selects OBSTACLES. Returns False for malformed/nonfinite paths.
    """
    path = np.asarray(path)
    if path.ndim != 2 or path.shape[1] != 6 or not len(path) or not np.isfinite(path).all():
        return False
    return not any(robot_edge_in_collision(a, b, interpolate, collision, obstacles, max_joint_step)
                   for a, b in zip(path, np.vstack((path[1:], path[-1:]))))


def validate_refined_path(path, candidate, interpolate, collision):
    """Return candidate (M, 6) if endpoints and dense checks pass, otherwise None.

    path is the original (N, 6) path. interpolate and collision are the student's
    callbacks. Reject malformed/nonfinite outputs and endpoint changes beyond
    1e-6 radians before checking collisions.
    """
    if candidate is None:
        return None
    try:
        candidate = np.asarray(candidate)
    except ValueError:
        return None
    if (candidate.ndim != 2 or candidate.shape[1] != 6 or not len(candidate)
            or not np.issubdtype(candidate.dtype, np.number)
            or not np.isrealobj(candidate) or not np.isfinite(candidate).all()):
        return None
    if not np.allclose(candidate[[0, -1]], path[[0, -1]], atol=1e-6, rtol=0):
        return None
    if not path_is_valid(candidate, interpolate, collision):
        return None
    return candidate


def prepare_trajopt(path):
    """Build a TrajOpt request using joint path (N, 6) as the initial guess.

    Returns a dictionary containing the planner, request, and robot owner.
    Only endpoints are constrained; intermediate waypoints are free to move.
    This loads the optional Tesseract dependency only when called.
    """
    if np.all(path == path[0]):
        return {"constant": path.copy()}
    from tesseract_robotics.planning import Robot, Pose, create_obstacle, sphere
    from tesseract_robotics.planning.profiles import create_trajopt_default_profiles
    from tesseract_robotics.tesseract_common import ManipulatorInfo
    from tesseract_robotics.tesseract_command_language import (
        CompositeInstruction, JointWaypoint, JointWaypointPoly_wrap_JointWaypoint,
        MoveInstruction, MoveInstructionType_FREESPACE, MoveInstructionPoly_wrap_MoveInstruction,
    )
    from tesseract_robotics.tesseract_motion_planners import PlannerRequest
    from tesseract_robotics.tesseract_motion_planners_trajopt import TrajOptMotionPlanner

    robot = Robot.from_files(str(ASSETS / "robot_spheres.urdf"), str(ASSETS / "robot.srdf"))
    for index, obstacle in enumerate(OBSTACLES):
        create_obstacle(robot, name=f"obstacle_{index}", geometry=sphere(float(obstacle[3])),
                        transform=Pose.from_xyz(*obstacle[:3]))
    robot.set_joints(path[0], JOINT_NAMES)
    program = CompositeInstruction("DEFAULT")
    info = ManipulatorInfo()
    info.manipulator, info.working_frame, info.tcp_frame = "manipulator", "base_link", "ee_link"
    program.setManipulatorInfo(info)
    for index, theta in enumerate(path):
        waypoint = JointWaypoint(JOINT_NAMES, np.asarray(theta, dtype=float))
        waypoint.setIsConstrained(index in (0, len(path) - 1))
        move = MoveInstruction(JointWaypointPoly_wrap_JointWaypoint(waypoint),
                               MoveInstructionType_FREESPACE, "DEFAULT")
        program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(move))
    request = PlannerRequest()
    request.env = robot.env
    request.instructions = program
    request.profiles = create_trajopt_default_profiles()
    request.format_result_as_input = False
    return {"planner": TrajOptMotionPlanner("TrajOptMotionPlannerTask"),
            "request": request, "robot": robot}


def solve_trajopt(problem):
    """Solve a problem from prepare_trajopt; return (N, 6) joints or None on failure.

    The returned path must still be checked independently for collisions.
    """
    if "constant" in problem:
        return problem["constant"].copy()
    from tesseract_robotics.tesseract_command_language import (
        InstructionPoly_as_MoveInstructionPoly, WaypointPoly_as_StateWaypointPoly,
        WaypointPoly_as_JointWaypointPoly,
    )
    response = problem["planner"].solve(problem["request"])
    if not response.successful:
        return None
    path = []
    for instruction in response.results:
        waypoint = InstructionPoly_as_MoveInstructionPoly(instruction).getWaypoint()
        waypoint = (WaypointPoly_as_StateWaypointPoly(waypoint) if waypoint.isStateWaypoint()
                    else WaypointPoly_as_JointWaypointPoly(waypoint))
        names = list(waypoint.getNames())
        path.append(np.asarray(waypoint.getPosition())[[names.index(name) for name in JOINT_NAMES]])
    return np.array(path)
