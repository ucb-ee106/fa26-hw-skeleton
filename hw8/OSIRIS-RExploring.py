"""SymPy starter for HW8: OSIRIS-RExploring, parts (b) and (c).

Run python hw8/OSIRIS-RExploring.py after completing the TODOs.
Documentation: https://docs.sympy.org/latest/index.html
"""

import sympy as sp


def main():
    """
    Enter the matrices from part (a) and construct the controllability matrix before running this script.
    """
    p = sp.symbols("p", positive=True)
    omega = sp.symbols("omega", real=True, nonzero=True)

    # TODO: enter A (4, 4) and B (4, 2) from part (a), using sp.Matrix.
    A = ...
    B = ...

    # (b) Each eigenvalue is paired with its algebraic multiplicity.
    print("Eigenvalues and algebraic multiplicities:")
    sp.pprint()

    # Each entry contains an eigenvalue, multiplicity, and eigenvector basis.
    print("Eigenvalues, algebraic multiplicities, and eigenvector bases:")
    sp.pprint(A.eigenvects())

    # (c) TODO: construct C = [B, AB, A**2 B, A**3 B], with shape (4, 8).
    # sp.Matrix.hstack joins matrices horizontally; * multiplies SymPy matrices.
    C = ...

    print("Controllability matrix:")
    sp.pprint(C)
    print("Rank:", C.rank())


if __name__ == "__main__":
    main()
