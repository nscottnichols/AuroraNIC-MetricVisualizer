import numpy as np
import pytest

from expressions import FormulaError, evaluate_formula


def test_accepts_benign_metric_math():
    values = np.array([1.0, 5.0, 9.0])
    result = evaluate_formula("(m[0]-m[1])/(m[2]+1)", values)

    np.testing.assert_allclose(result, -0.4)


def test_accepts_allowlisted_numpy_function():
    values = np.array([[4.0]])

    np.testing.assert_allclose(evaluate_formula("sqrt(m[0])", values), [2.0])


@pytest.mark.parametrize(
    "formula",
    [
        "m.__class__",
        "np.sqrt(m[0])",
        "__import__('os')",
        "open('/etc/passwd')",
        "(lambda value: value)(m[0])",
        "[value for value in m]",
        "m[0:1]",
        "m['metric']",
        "m[0] ** 999999",
        "unknown_function(m[0])",
        "sum(m, axis=0)",
        "m[0] and m[1]",
    ],
)
def test_rejects_unsafe_or_unsupported_expressions(formula):
    values = np.zeros((1, 1, 1))

    with pytest.raises(FormulaError):
        evaluate_formula(formula, values)


def test_rejects_oversized_input():
    values = np.zeros((1,))

    with pytest.raises(FormulaError, match="character limit"):
        evaluate_formula("m[0] + " + "0" * 300, values)


def test_rejects_resource_exhausting_expressions():
    values = np.zeros((1,))

    with pytest.raises(FormulaError, match="nesting depth"):
        evaluate_formula("+".join(["m[0]"] * 40), values)

    with pytest.raises(FormulaError, match="element limit"):
        evaluate_formula("m[0]", np.zeros((1_000_001,)))
