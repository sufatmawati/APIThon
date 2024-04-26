import ast
import math
from typing import Any, Callable, Union

VALID_CONST_TYPE = {int, float, str, bool}
VALID_BINOP_TYPE = {int, float}

FN_COMPREHENSION_INIT = {
    ast.ListComp: list,
    ast.SetComp: set,
    ast.DictComp: dict,
}


NumericType = Union[int, float]
NumericOrStringType = Union[int, float, str]

def binary_add(left: NumericOrStringType, right: NumericOrStringType) -> NumericOrStringType:
    return left + right


def binary_sub(left: NumericType, right: NumericType) -> NumericType:
    return left - right


def binary_mult(left: NumericType, right: NumericType) -> NumericType:
    return left * right


def binary_div(left: NumericType, right: NumericType) -> NumericType:
    return left / right


def binary_floor_div(left: NumericType, right: NumericType) -> NumericType:
    return left // right


def binary_mod(left: NumericType, right: NumericType) -> NumericType:
    return left % right


def binary_pow(left: NumericType, right: NumericType) -> NumericType:
    return math.pow(left, right)

def boolean_and(left: Any, right: Any) -> Any:
    return left and right


def boolean_or(left: Any, right: Any) -> Any:
    return left or right

def comparison_eq(left: Any, right: Any) -> bool:
    return left == right


def comparison_not_eq(left: Any, right: Any) -> bool:
    return left != right


def comparison_less_than(left: Any, right: Any) -> bool:
    return left < right


def comparison_greater_than(left: Any, right: Any) -> bool:
    return left > right


def comparison_less_or_equal(left: Any, right: Any) -> bool:
    return left <= right


def comparison_greater_or_equal(left: Any, right: Any) -> bool:
    return left >= right


def comparison_is(left: Any, right: Any) -> bool:
    return left is right


def comparison_is_not(left: Any, right: Any) -> bool:
    return left is not right


def comparison_in(left: Any, right: Any) -> bool:
    return left in right


def comparison_not_in(left: Any, right: Any) -> bool:
    return left not in right

def comprehension_multi_dict(
    execute_node_fn: Callable[[ast.DictComp], Any], target: dict, node: ast.AST
):
    target.update(
        execute_node_fn(
            ast.DictComp(key=node.key, value=node.value, generators=node.generators[1:])
        )
    )


def comprehension_multi_list(
    execute_node_fn: Callable[[ast.ListComp], Any], target: list, node: ast.AST
):
    target.extend(execute_node_fn(ast.ListComp(elt=node.elt, generators=node.generators[1:])))


def comprehension_multi_set(
    execute_node_fn: Callable[[ast.SetComp], Any], target: set, node: ast.AST
):
    target.update(execute_node_fn(ast.SetComp(elt=node.elt, generators=node.generators[1:])))


def comprehension_single_dict(
    execute_node_fn: Callable[[ast.AST], Any], target: dict, node: ast.AST
):
    target.update({execute_node_fn(node.key): execute_node_fn(node.value)})


def comprehension_single_list(
    execute_node_fn: Callable[[ast.AST], Any], target: list, node: ast.AST
):
    target.append(execute_node_fn(node.elt))


def comprehension_single_set(execute_node_fn: Callable[[ast.AST], Any], target: set, node: ast.AST):
    target.add(execute_node_fn(node.elt))

def unary_not(value: Any) -> bool:
    return not value


def unary_add(value: NumericType) -> NumericType:
    return +value


def unary_sub(value: NumericType) -> NumericType:
    return -value


def unary_invert(value: int) -> int:
    return ~value


FN_BINARY_OP = {
    ast.Add: binary_add,
    ast.Div: binary_div,
    ast.FloorDiv: binary_floor_div,
    ast.Mod: binary_mod,
    ast.Mult: binary_mult,
    ast.Sub: binary_sub,
    ast.Pow: binary_pow,
}

FN_BOOLEAN_OP = {
    ast.And: (boolean_and, True),
    ast.Or: (boolean_or, False),
}

FN_COMPARISON_OP = {
    ast.Eq: comparison_eq,
    ast.Gt: comparison_greater_than,
    ast.GtE: comparison_greater_or_equal,
    ast.In: comparison_in,
    ast.Is: comparison_is,
    ast.IsNot: comparison_is_not,
    ast.Lt: comparison_less_than,
    ast.LtE: comparison_less_or_equal,
    ast.NotEq: comparison_not_eq,
    ast.NotIn: comparison_not_in,
}

FN_COMPREHENSION_SINGLE = {
    ast.DictComp: comprehension_single_dict,
    ast.ListComp: comprehension_single_list,
    ast.SetComp: comprehension_single_set,
}

FN_COMPREHENSION_MULTI = {
    ast.DictComp: comprehension_multi_dict,
    ast.ListComp: comprehension_multi_list,
    ast.SetComp: comprehension_multi_set,
}

FN_UNARY_OP = {
    ast.Invert: unary_invert,
    ast.Not: unary_not,
    ast.UAdd: unary_add,
    ast.USub: unary_sub,
}
