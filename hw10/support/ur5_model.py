"""Provided symbolic geometry for the same UR5 description used in HW8."""

from math import pi
from pathlib import Path
import xml.etree.ElementTree as ET

import sympy as sp


DESCRIPTION = Path(__file__).resolve().parents[2] / "hw8/planning/assets/robot_visual.urdf"
GRAVITY = sp.Matrix([0, 0, -sp.Rational("9.81")])


def _vector(text):
    return sp.Matrix([sp.Rational(value) for value in text.split()])


def _rotation(axis, angle):
    x, y, z = axis
    skew = sp.Matrix([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return sp.eye(3) + sp.sin(angle) * skew + (1 - sp.cos(angle)) * skew**2


def _origin(element):
    position = _vector(element.get("xyz", "0 0 0"))
    # This URDF stores its right angles as decimals. Keep them exact in SymPy.
    roll, pitch, yaw = [round(2 * float(value) / pi) * sp.pi / 2
                        for value in element.get("rpy", "0 0 0").split()]
    rotation = _rotation((0, 0, 1), yaw) @ _rotation((0, 1, 0), pitch)
    return position, rotation @ _rotation((1, 0, 0), roll)


def ur5_parameters():
    """Return home twists (6, 6), six COM poses, six link inertias, and masses.

    All quantities are exact SymPy objects in HW8 joint order. The link
    inertias are (6, 6), expressed at each center of mass, in [v; omega]
    order. This data comes from the same bare UR5 used by the viewer.
    """
    if not DESCRIPTION.is_file():
        raise FileNotFoundError("HW10 uses the UR5 description from HW8. Restore the hw8 folder.")
    description = ET.parse(DESCRIPTION).getroot()
    twists, home_poses, inertias, masses = [], [], [], []
    position, rotation = sp.zeros(3, 1), sp.eye(3)
    links = {link.get("name"): link for link in description.findall("link")}

    for joint in description.findall("joint[@type='revolute']"):
        offset, orientation = _origin(joint.find("origin"))
        position = position + rotation @ offset
        rotation = rotation @ orientation
        axis = rotation @ _vector(joint.find("axis").get("xyz"))
        twists.append(position.cross(axis).col_join(axis))
        inertial = links[joint.find("child").get("link")].find("inertial")
        com, orientation = _origin(inertial.find("origin"))
        home = (rotation @ orientation).row_join(position + rotation @ com)
        home_poses.append(home.col_join(sp.Matrix([[0, 0, 0, 1]])))
        inertia = inertial.find("inertia")
        tensor = sp.Matrix(3, 3, lambda i, j: sp.Rational(
            inertia.get("i" + "xyz"[min(i, j)] + "xyz"[max(i, j)])
        ))
        mass = sp.Rational(inertial.find("mass").get("value"))
        masses.append(mass)
        inertias.append(sp.diag(mass * sp.eye(3), tensor))
    return sp.Matrix.hstack(*twists), home_poses, inertias, masses


def tool_home_pose():
    """Return the exact (4, 4) tool0 pose at zero joint angles."""
    description = ET.parse(DESCRIPTION).getroot()
    position, rotation = sp.zeros(3, 1), sp.eye(3)
    joints = description.findall("joint[@type='revolute']")
    joints.append(description.find("joint[@name='wrist_3_link-tool0_fixed_joint']"))
    for joint in joints:
        offset, orientation = _origin(joint.find("origin"))
        position += rotation @ offset
        rotation = rotation @ orientation
    return rotation.row_join(position).col_join(sp.Matrix([[0, 0, 0, 1]]))
