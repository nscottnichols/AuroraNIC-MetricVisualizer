"""Bounded expression evaluation for user-defined calculated metrics."""

import ast

import numpy as np


class FormulaError(ValueError):
    """Raised when a calculated-metric formula is invalid or unsafe."""


MAX_FORMULA_LENGTH = 256
MAX_AST_NODES = 64
MAX_AST_DEPTH = 32
MAX_CALL_ARGUMENTS = 3
MAX_FORMULA_INPUT_ELEMENTS = 1_000_000

_ALLOWED_FUNCTIONS = {
    "abs": np.abs,
    "clip": np.clip,
    "exp": np.exp,
    "isfinite": np.isfinite,
    "isinf": np.isinf,
    "isnan": np.isnan,
    "log": np.log,
    "log10": np.log10,
    "log2": np.log2,
    "max": np.maximum,
    "maximum": np.maximum,
    "mean": np.mean,
    "median": np.median,
    "min": np.minimum,
    "minimum": np.minimum,
    "nanmean": np.nanmean,
    "nanstd": np.nanstd,
    "nanvar": np.nanvar,
    "sqrt": np.sqrt,
    "std": np.std,
    "sum": np.sum,
    "var": np.var,
    "where": np.where,
}

_ALLOWED_BINARY_OPERATORS = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
)
_ALLOWED_UNARY_OPERATORS = (ast.UAdd, ast.USub)


def evaluate_formula(formula, values):
    """Evaluate a short, allowlisted numeric expression against ``values``.

    The only variable is ``m``. Subscripts must use literal integer indices,
    function calls must use the fixed function allowlist, and attribute access,
    imports, lambdas, comprehensions, keyword arguments, and exponentiation are
    rejected before evaluation.

    Allowed operations return either a scalar or an elementwise broadcast with
    the input shape; they cannot construct a larger array. Together with the
    input-element, node-count, depth, and length limits, this bounds formula
    work and intermediate memory relative to the selected metric array.
    """
    if not isinstance(formula, str):
        raise FormulaError("Formula must be text")
    if not formula.strip():
        raise FormulaError("Formula cannot be empty")
    if len(formula) > MAX_FORMULA_LENGTH:
        raise FormulaError(
            f"Formula exceeds the {MAX_FORMULA_LENGTH}-character limit"
        )
    if not isinstance(values, np.ndarray):
        raise FormulaError("Metric values must be a NumPy array")
    if values.size > MAX_FORMULA_INPUT_ELEMENTS:
        raise FormulaError(
            f"Metric selection exceeds the {MAX_FORMULA_INPUT_ELEMENTS:,}-element limit"
        )

    try:
        tree = ast.parse(formula, mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        raise FormulaError(f"Invalid formula syntax: {exc}") from exc

    node_count = 0

    def evaluate(node, depth):
        nonlocal node_count
        node_count += 1
        if node_count > MAX_AST_NODES:
            raise FormulaError(f"Formula exceeds {MAX_AST_NODES} expression nodes")
        if depth > MAX_AST_DEPTH:
            raise FormulaError(f"Formula exceeds nesting depth {MAX_AST_DEPTH}")

        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(
                node.value, (int, float)
            ):
                raise FormulaError("Only numeric constants are allowed")
            return node.value

        if isinstance(node, ast.Name):
            if node.id != "m":
                raise FormulaError(
                    "Only the metric variable 'm' is allowed"
                )
            return values

        if isinstance(node, ast.BinOp):
            if type(node.op) not in _ALLOWED_BINARY_OPERATORS:
                raise FormulaError(
                    "Operator is not allowed; exponentiation is disabled"
                )
            left = evaluate(node.left, depth + 1)
            right = evaluate(node.right, depth + 1)
            return _apply_operator(type(node.op), left, right)

        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in _ALLOWED_UNARY_OPERATORS:
                raise FormulaError("Unary operator is not allowed")
            operand = evaluate(node.operand, depth + 1)
            return +operand if isinstance(node.op, ast.UAdd) else -operand

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise FormulaError("Only direct allowlisted function calls are allowed")
            function_name = node.func.id
            if function_name not in _ALLOWED_FUNCTIONS:
                raise FormulaError(f"Function '{function_name}' is not allowed")
            if node.keywords:
                raise FormulaError("Keyword arguments are not allowed")
            if len(node.args) > MAX_CALL_ARGUMENTS:
                raise FormulaError(
                    f"Function calls may have at most {MAX_CALL_ARGUMENTS} arguments"
                )
            arguments = [evaluate(argument, depth + 1) for argument in node.args]
            return _ALLOWED_FUNCTIONS[function_name](*arguments)

        if isinstance(node, ast.Subscript):
            if not isinstance(node.value, ast.Name) or node.value.id != "m":
                raise FormulaError("Subscripts are only allowed on the variable 'm'")
            index_node = node.slice
            if hasattr(ast, "Index") and isinstance(index_node, ast.Index):
                index_node = index_node.value
            if not isinstance(index_node, ast.Constant):
                raise FormulaError("Metric subscripts must be literal integers")
            if isinstance(index_node.value, bool) or not isinstance(index_node.value, int):
                raise FormulaError("Metric subscripts must be literal integers")
            try:
                return values[index_node.value]
            except (IndexError, TypeError) as exc:
                raise FormulaError(
                    f"Metric subscript {index_node.value} is out of range"
                ) from exc

        raise FormulaError("Formula contains an unsupported expression")

    return evaluate(tree.body, 0)


def _apply_operator(operator, left, right):
    if operator is ast.Add:
        return left + right
    if operator is ast.Sub:
        return left - right
    if operator is ast.Mult:
        return left * right
    if operator is ast.Div:
        return left / right
    if operator is ast.FloorDiv:
        return left // right
    if operator is ast.Mod:
        return left % right
    raise FormulaError("Operator is not allowed")
