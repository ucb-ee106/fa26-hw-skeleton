"""Check HW10 functions against saved public cases, without running Swift."""

from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import sympy as sp


CASES = json.loads((Path(__file__).parent / "support/hw10_cases.json").read_text())
Q, QD = sp.symbols("q:6"), sp.symbols("v:6")
SYMBOLS = {str(symbol): symbol for symbol in (*Q, *QD)}


def load_submission(path):
    """Load hw10.py so the public cases can call the submitted functions."""
    spec = importlib.util.spec_from_file_location("hw10_submission", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {spec.name: module}):
        spec.loader.exec_module(module)
    return module


def _matrix(rows):
    return sp.Matrix(rows).applyfunc(lambda entry: sp.sympify(entry, locals=SYMBOLS))


def _argument(argument):
    if not isinstance(argument, dict):
        return argument
    kind, data = next(iter(argument.items()))
    if kind == "array":
        return np.array(data, dtype=float)
    if kind == "matrix":
        return _matrix(data)
    if kind == "matrices":
        return [_matrix(rows) for rows in data]
    if kind == "symbols":
        return tuple(SYMBOLS[name] for name in data)
    if kind == "link_jacobian":
        case_twists = _matrix(data["twists"])
        samples = {sample["link_number"]: sample for sample in data["samples"]}

        def lookup(theta, twists, home_pose, link_number):
            sample = samples.get(link_number)
            if (sample is not None and twists == case_twists and tuple(theta) == Q
                    and home_pose == _matrix(sample["home_pose"])):
                return _matrix(sample["result"])
            raise ValueError("Unexpected arguments to link_jacobian")

        return lookup
    if kind == "dynamics":
        def dynamics(q, qd):
            for sample in data:
                if np.allclose(q, sample["q"]) and np.allclose(qd, sample["qd"]):
                    return tuple(np.array(value, dtype=float) for value in sample["model"])
            raise ValueError(f"Unexpected dynamics evaluation at q={q}, qd={qd}")

        return dynamics
    raise ValueError(f"Unknown fixture kind {kind}")


def _matches(actual, expected):
    array = np.asarray(actual)
    target = np.asarray(expected)
    return (array.shape == target.shape and array.dtype.kind in "iuf"
            and np.all(np.isfinite(array))
            and np.allclose(array, target, rtol=1e-7, atol=1e-9))


def evaluate_case(implementation, case):
    """Run one saved case and return (passed, actual) for the checker and grader."""
    function = case["function"]
    expected = case["expected"]
    try:
        with ExitStack() as bindings:
            for name, data in case.get("helpers", {}).items():
                bindings.enter_context(patch.object(implementation, name, _argument({name: data}), create=True))
            actual = getattr(implementation, function)(*map(_argument, case["args"]))
        if function in ("link_jacobian", "inertia_matrix", "coriolis_matrix", "gravity_vector"):
            if not isinstance(actual, sp.MatrixBase):
                return False, actual
            substitutions = {SYMBOLS[name]: value for name, value in case["substitute"].items()}
            actual = np.array(actual.subs(substitutions).evalf(), dtype=float)
        return bool(_matches(actual, expected)), actual
    except (Exception, SystemExit) as error:
        return False, f"{type(error).__name__}: {error}"


def case_detail(case, passed, actual):
    """Format the supplied inputs, expected output, and actual output of a case."""
    inputs = {key: value for key, value in case.items() if key not in ("name", "expected")}
    return (f"[{'PASS' if passed else 'FAIL'}] {case['name']}\n"
            f"Test case: {inputs!r}\nExpected: {case['expected']!r}\nActual: {actual!r}")


def main():
    """Print all public checks for the student's hw10.py and return an exit status."""
    hw10 = load_submission(Path(__file__).with_name("hw10.py"))

    passed_count = 0
    for case in CASES:
        passed, actual = evaluate_case(hw10, case)
        passed_count += passed
        if passed:
            print(f"[PASS] {case['name']}")
        else:
            print(case_detail(case, passed, actual))
    print(f"{passed_count}/{len(CASES)} HW10 checks passed.")
    return int(passed_count != len(CASES))


if __name__ == "__main__":
    raise SystemExit(main())
