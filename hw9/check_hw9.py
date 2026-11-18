"""Check your PID implementation against the public HW9 cases."""

import json
from pathlib import Path

import numpy as np


CASES = json.loads(Path(__file__).with_name("hw9_cases.json").read_text())


def evaluate_case(module, case):
    """Return (passed, actual) for one case against its saved expected values."""
    actual = []
    try:
        kind = case["kind"]
        if kind == "finite_difference":
            actual.append(module.finite_difference(*case["args"]))
        else:
            if kind == "filter":
                update = module.LowPassFilter(**case["kwargs"]).update
            else:
                update = module.PIDController(**case["kwargs"]).update
            for args in case["samples"]:
                actual.append(update(*args))
        values = np.asarray(actual)
        expected = np.asarray(case["expected"])
        passed = (values.shape == expected.shape and values.dtype.kind in "iuf"
                  and np.all(np.isfinite(values))
                  and np.allclose(values, expected, rtol=1e-7, atol=1e-9))
        return bool(passed), actual
    except (Exception, SystemExit) as error:
        actual.append(f"{type(error).__name__}: {error}")
        return False, actual


def case_detail(case, passed, actual):
    """Format the case name, inputs, expected output, and actual output."""
    inputs = {key: value for key, value in case.items() if key not in ("name", "expected")}
    return (f"[{'PASS' if passed else 'FAIL'}] {case['name']}\n"
            f"Test case: {inputs!r}\nExpected: {case['expected']!r}\nActual: {actual!r}")


def main():
    """Print every public case for the local pid.py and return a process status."""
    import pid

    count = passed_count = 0
    for cases in CASES.values():
        for case in cases:
            passed, actual = evaluate_case(pid, case)
            passed_count += passed
            count += 1
            print(case_detail(case, passed, actual))
    print(f"{passed_count}/{count} HW9 checks passed.")
    return int(passed_count != count)


if __name__ == "__main__":
    raise SystemExit(main())
