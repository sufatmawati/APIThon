import ast
from dataclasses import dataclass, field
from types import (
    BuiltinFunctionType,
    BuiltinMethodType,
    FunctionType,
    GeneratorType,
    LambdaType,
    MethodType,
)
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Set, Tuple, Union

from apithon_impl import (
    FN_BINARY_OP,
    FN_BOOLEAN_OP,
    FN_COMPARISON_OP,
    FN_COMPREHENSION_INIT,
    FN_COMPREHENSION_MULTI,
    FN_COMPREHENSION_SINGLE,
    FN_UNARY_OP,
    VALID_BINOP_TYPE,
    VALID_CONST_TYPE,
)
from exceptions import (
    Break,
    Continue,
    MaximumOperationExceededError,
    MaximumSizeExceededError,
    PrivateAccessError,
    Return,
)

_AST_TYPE_TO_DISPLAY_NAME = {
    ast.Add: 'addition operator',
    ast.Assign: 'variable(s) assignement',
    ast.Attribute: 'attribute access',
    ast.AugAssign: 'augmented assignment',
    ast.BinOp: 'binary operation',
    ast.BitAnd: 'bitwise and operator',
    ast.BitOr: 'bitwise or operator',
    ast.BitXor: 'bitwise exclusive or operator',
    ast.BoolOp: 'boolean operation',
    ast.Break: 'break statement',
    ast.Call: 'function call',
    ast.Compare: 'comparison of two or more values',
    ast.Constant: 'constant value',
    ast.Continue: 'continue statement',
    ast.Delete: 'delete statement',
    ast.Dict: 'dictionary object',
    ast.DictComp: 'dictionary comprehension',
    ast.Div: 'division operator',
    ast.Eq: 'equal comparison',
    ast.Expr: 'standalone expression',
    ast.FloorDiv: 'integer division operator',
    ast.For: 'for loop',
    ast.FormattedValue: 'formatting field in an f-string',
    ast.FunctionDef: 'function definition',
    ast.GeneratorExp: 'fenerator expression',
    ast.Gt: 'greater than comparison',
    ast.GtE: 'greater or equal comparison',
    ast.If: 'if statement',
    ast.IfExp: 'inline if-expression',
    ast.In: 'in comparison',
    ast.Is: 'is comparison',
    ast.IsNot: 'is not comparison',
    ast.JoinedStr: 'f-string',
    ast.Lambda: 'lambda expression',
    ast.List: 'list object',
    ast.ListComp: 'list comprehension',
    ast.LShift: 'left shift operator',
    ast.Lt: 'less than comparison',
    ast.LtE: 'less or equal comparison',
    ast.Mod: 'modulo operator',
    ast.Mult: 'multiplication operator',
    ast.Name: 'variable name',
    ast.NamedExpr: 'named "walrus" assignment',
    ast.NotEq: 'not equal comparison',
    ast.NotIn: 'not in comparison',
    ast.Pow: 'power operator',
    ast.Return: 'return statement',
    ast.RShift: 'right shift operator',
    ast.Set: 'set object',
    ast.SetComp: 'set comprehension',
    ast.Starred: 'starred *var variable reference',
    ast.Sub: 'subtraction operator',
    ast.Subscript: 'subscript access',
    ast.Tuple: 'tuple object',
    ast.UnaryOp: 'unary operation',
    ast.While: 'while loop',
}

_AST_TYPE_TO_OPERATOR = {
    ast.Add: '+',
    ast.Sub: '-',
    ast.Mult: '*',
    ast.Div: '/',
    ast.FloorDiv: '//',
    ast.Mod: '%',
    ast.Pow: '**',
    ast.LShift: '<<',
    ast.RShift: '>>',
    ast.BitOr: '|',
    ast.BitXor: '^',
    ast.BitAnd: '&',
}


def _display(value):
    t = type(value)
    return _AST_TYPE_TO_DISPLAY_NAME.get(t, t.__name__)


@dataclass
class ExecutionResult:
    """Class for keeping track of an item in inventory."""

    return_value: Any = None
    print_log: List[str] = field(default_factory=list)
    execution_log: List[str] = field(default_factory=list)
    err: Optional[Exception] = None
    line: Optional[str] = None


class MemoryStack:
    def __init__(self):
        self._mem = [{}]

    def update(self, entries):
        self._mem[-1].update(entries)

    def push(self, mem_dict=None):
        self._mem.append(mem_dict or {})

    def pop(self, mem_index=-1):
        ret = self._mem[mem_index]
        self._mem = self._mem[:mem_index]
        return ret

    def find(self, name: str):
        # want to iterate from most local context to least local
        for shard in reversed(self._mem):
            if name in shard:
                return shard[name]
        raise RuntimeError(f'name \'{name}\' is not defined')

    def delete(self, name: str):
        # want to iterate from most local context to least local
        for shard in reversed(self._mem):
            if name in shard:
                del shard[name]
                return None
        raise RuntimeError(f'name \'{name}\' is not defined')

    def __len__(self):
        return len(self._mem)

    def __iter__(self):
        return iter(self._mem)

    def __str__(self):
        return str(self._mem)


class APIthonCodeExecuter:
    """APIythonCodeExecuter safely executes GPT generated code with the following guarantees.

    1. Only functions/objects passed to the executer in the constructor can be called or used at runtime.
    Non lambda function definition and class definition is not allowed. These are the only functions that can
    have side affects. This means the security of the executed code is dependant on the security of the provided
    functions. Before adding a function, consider the consquences of giving an adverserial user access
    to the function.
    2. This executer has its own memory, and does not have access to anything in the normal python namespace
    unless one the functions/objects explicity passed to the executer has access to the global namespace
    3. Access to private variables and methods (preceeded by _) are not allowed in this language
    4. Limits on integer and iterable length to prevent running out of memory
    5. Limit on the number of operations to prevent programs that run infinitely
    """

    _FEATURE_VERSION = (3, 8)  # Python 3.8

    def __init__(
        self,
        supported_functions: Dict[str, Callable] = None,
        debug=False,
        max_constant=4294967296,
        min_constant=-4294967296,
        max_string_length=4096,
        max_iterable_length=2048,
        max_operations=8192,
        max_validation_size=8192,
        max_code_size=4096,
    ):
        self._max_constant = max_constant
        self._min_constant = min_constant
        self._max_string_length = max_string_length
        self._max_iterable_length = max_iterable_length
        self._max_operations = max_operations
        self._max_validation_size = max_validation_size
        self._max_code_size = max_code_size

        self.supported_functions = supported_functions if supported_functions is not None else {}
        self.supported_functions.update(
            {
                'print': self.print,
                'range': self.range,
                'map': self.map,
                'filter': self.filter,
                'len': len,
                'all': all,
                'any': any,
                'sum': sum,
                'str': str,
                'int': int,
                'float': float,
                'set': set,
                'dict': dict,
                'list': list,
                'max': max,
                'next': next,
                'min': min,
                'bool': bool,
                'round': round,
                'sorted': sorted,
                'reversed': reversed,
            }
        )
        self.supported_primitives_types = {
            BuiltinMethodType,
            BuiltinFunctionType,
            LambdaType,
            int,
            float,
            set,
            dict,
            list,
            str,
            bool,
            range,
            GeneratorType,
            reversed,
        }
        self.supported_primitives_types.update(
            [
                value if callable(value) else type(value)
                for value in self.supported_functions.values()
            ]
        )
        self.ast_type_to_executor = {
            ast.Constant: self.__execute_constant,
            ast.List: self.__execute_list,
            ast.Set: self.__execute_set,
            ast.Dict: self.__execute_dict,
            ast.Expr: self.__execute_expr,
            ast.UnaryOp: self.__execute_unaryop,
            ast.BinOp: self.__execute_binop,
            ast.BoolOp: self.__execute_boolop,
            ast.Compare: self.__execute_compare,
            ast.IfExp: self.__execute_ifexp,
            ast.Attribute: self.__execute_attribute,
            ast.Name: self.__execute_name,
            ast.NamedExpr: self.__execute_namedexpr,
            ast.Subscript: self.__execute_subscript,
            ast.AugAssign: self.__execute_augassign,
            ast.Assign: self.__execute_assign,
            ast.If: self.__execute_if,
            ast.For: self.__execute_for,
            ast.Break: self.__execute_break,
            ast.Continue: self.__execute_continue,
            ast.Return: self.__execute_return,
            ast.Call: self.__execute_call,
            ast.Starred: self.__execute_starred,
            ast.Delete: self.__execute_delete,
            ast.Tuple: self.__execute_tuple,
            ast.ListComp: self.__execute_listcomp,
            ast.Lambda: self.__execute_lambda,
            ast.JoinedStr: self.__execute_joinedstr,
            ast.SetComp: self.__execute_setcomp,
            ast.DictComp: self.__execute_dictcomp,
            ast.While: self.__execute_while,
            ast.FormattedValue: self.__execute_formatted_value,
            ast.GeneratorExp: self.__execute_generator_exp,
            ast.FunctionDef: self.__execute_function_def,
        }
        self.size_validator = SizeValidator(
            self.supported_primitives_types,
            max_validation_size=max_validation_size,
            max_string_length=max_string_length,
            max_iterable_length=max_iterable_length,
            max_constant=max_constant,
            min_constant=min_constant,
        )
        self.reset()
        self.debug = debug

    def print(self, *args: List[Any]) -> None:
        self.print_log.extend([str(arg) for arg in args])

    def inc_operations(self, i: int = 1) -> None:
        self.num_operation += i
        if self.num_operation > self._max_operations:
            raise MaximumOperationExceededError('Program exceeded maximum allowed runtime')

    def range(self, i: int) -> Iterable:
        it = 0
        while it < i:
            self.inc_operations()
            yield it
            it += 1

    def map(self, fn: Callable, iterable: Iterable, *iterables: Iterable[Iterable]) -> Iterable:
        if len(iterables) == 0:
            for el in iterable:
                self.inc_operations()
                yield fn(el)
        else:
            iterables = [iter(iterable)] + [iter(it) for it in iterables]
            while True:
                try:
                    args = [next(it) for it in iterables]
                    self.inc_operations()
                    yield fn(*args)
                except StopIteration:
                    return

    def filter(self, fn: Callable, iterable: Iterable) -> Iterable:
        for el in iterable:
            self.inc_operations()
            if fn(el):
                yield el

    def reset(self):
        """Resets the memory of the apithon code executer."""
        self.num_operation = 0
        self.print_log = []
        self.execution_log = []
        self.mem = MemoryStack()
        self.mem.update(self.supported_functions)
        self.mem.push()

    def __add_to_execution_log(self, log: str):
        if self.debug:
            self.execution_log.append(log)

    def node_eval(self, node: ast.AST) -> Any:
        return self.__execute_node(node)

    def execute_code(self, code: str) -> ExecutionResult:
        to_return = None
        self.num_operation = 0
        try:
            if len(code) > self._max_code_size:
                raise MaximumSizeExceededError(f'code length exceeds maximum {self._max_code_size}')
            parsed_ast = ast.parse(code, feature_version=self._FEATURE_VERSION)
        except Exception as e:
            return ExecutionResult(err=e, line=code)
        for i, node in enumerate(parsed_ast.body):
            code_line = ast.get_source_segment(code, node)
            try:
                out = self.__execute_node(node)
                if i == len(parsed_ast.body) - 1:
                    to_return = out
                    break
                self.size_validator.validate_memory(self.mem)
            except Return as e:
                to_return = e.value
                break
            except Exception as e:
                return ExecutionResult(
                    print_log=self.print_log,
                    execution_log=self.execution_log,
                    err=e,
                    line=code_line,
                )
        return ExecutionResult(
            return_value=to_return, print_log=self.print_log, execution_log=self.execution_log
        )

    def __execute_node_list(self, nodes: ast.AST) -> None:
        for node in nodes:
            self.__execute_node(node)
        return

    def __execute_node(self, node: ast.AST) -> Any:
        self.inc_operations()

        if node is None:
            return None

        if type(node) not in self.ast_type_to_executor:
            raise NotImplementedError(f'Forbidden operation: {_display(node)}')

        out = self.ast_type_to_executor[type(node)](node)
        self.size_validator.validate_size(out, set())
        return out

    def __execute_constant(self, node: ast.Constant) -> Union[int, float, str, bool, None]:
        value = node.value
        if value is None or type(value) in VALID_CONST_TYPE:
            return value
        raise NotImplementedError(f'{_display(value)} constant value not supported')

    def __execute_list(self, node: ast.List) -> List:
        elts = []
        for elt in node.elts:
            val = self.__execute_node(elt)
            elts.append(val)
        return elts

    def __execute_tuple(self, node: ast.Tuple) -> Tuple:
        return tuple(self.__execute_list(ast.List(elts=node.elts, ctx=node.ctx)))

    def __execute_set(self, node: ast.Set) -> Set:
        elts = set()
        for elt in node.elts:
            val = self.__execute_node(elt)
            elts.add(val)
        return elts

    def __execute_dict(self, node: ast.Dict) -> Dict:
        node_dict = {}
        for keynode, valuenode in zip(node.keys, node.values):
            key = self.__execute_node(keynode)
            value = self.__execute_node(valuenode)
            node_dict[key] = value
        return node_dict

    def __execute_expr(self, node: ast.Expr) -> Any:
        return self.__execute_node(node.value)

    def __execute_unaryop(self, node: ast.UnaryOp) -> Union[int, float, str, bool, None]:
        op = node.op
        value = self.__execute_node(node.operand)
        fn = FN_UNARY_OP.get(type(node.op))
        if fn is None:
            raise NotImplementedError(f'unary operator not supported: {_display(op)}')
        return fn(value)

    def __execute_binop(self, node: ast.BinOp) -> Union[int, float, str, bool, List, None]:
        left = self.__execute_node(node.left)
        right = self.__execute_node(node.right)

        if isinstance(left, str) or isinstance(right, str):
            if isinstance(node.op, ast.Add):
                self.__add_to_execution_log(f'performing "{left}" + "{right}"')
                return left + right
            else:
                raise NotImplementedError(
                    'Addition is the only binary operation allowed for strings'
                )

        if type(left) not in VALID_BINOP_TYPE or type(right) not in VALID_BINOP_TYPE:
            raise NotImplementedError(f'{_display(node.op)} only allowed for numerical values')

        self.__add_to_execution_log(
            f'performing {left} {_AST_TYPE_TO_OPERATOR.get(type(node.op), type(node.op).__name__)} {right}'
        )
        fn = FN_BINARY_OP.get(type(node.op))
        if fn is None:
            raise NotImplementedError(f'{_display(node.op)} not supported')
        return fn(left, right)

    def __execute_boolop(self, node: ast.BoolOp) -> bool:
        if len(node.values) == 0:
            raise NotImplementedError('boolean operations with no values are not supported')

        bool_val = self.__execute_node(node.values[0])

        fn, is_and = FN_BOOLEAN_OP.get(type(node.op))
        if fn is None:
            raise NotImplementedError(f'boolean operator not supported: {_display(node.op)}')

        for value in node.values[1:]:
            # Boolean expression early exit
            if is_and and not bool_val:
                return False
            elif not is_and and bool_val:
                return True

            val = self.__execute_node(value)
            bool_val = fn(bool_val, val)

        return bool_val

    def __execute_compare(self, node: ast.Compare) -> bool:
        left_compare = self.__execute_node(node.left)

        for op, right_compare_node in zip(node.ops, node.comparators):
            right_compare = self.__execute_node(right_compare_node)

            fn = FN_COMPARISON_OP.get(type(op))
            if fn is None:
                raise NotImplementedError(f'comparison operator not supported: {_display(op)}')

            left_compare = fn(left_compare, right_compare)

        return left_compare

    def __execute_ifexp(self, node: ast.IfExp) -> Any:
        test = self.__execute_node(node.test)
        if test:
            return self.__execute_node(node.body)
        return self.__execute_node(node.orelse)

    def __execute_attribute(self, node: ast.Attribute) -> Any:
        if node.attr.startswith('_'):
            raise PrivateAccessError(f'Attempting to access private name {node.attr}')

        obj = self.__execute_node(node.value)
        if isinstance(node.ctx, ast.Store):

            def update_slice(data):
                self.__add_to_execution_log(f'updating {_display(obj)} = {data}')
                setattr(obj, node.attr, data)
                return

            return update_slice
        value = getattr(obj, node.attr, None)
        if value is None:
            raise RuntimeError(f'Failed to access {node.attr} of {ast.dump(node)}')
        return value

    def __execute_name(self, node: ast.Name) -> str:
        var_id = node.id
        ctx = node.ctx

        if isinstance(ctx, ast.Load):
            return self.mem.find(var_id)
        if isinstance(ctx, ast.Store):
            return var_id
        if isinstance(ctx, ast.Del):
            return self.mem.delete(var_id)
        raise NotImplementedError(f'{type(ctx)} context not supported')

    def __execute_namedexpr(self, node: ast.NamedExpr) -> Any:
        target = self.__execute_node(node.target)
        value = self.__execute_node(node.value)
        self.__store(target, value)
        return value

    def __execute_subscript(self, node: ast.Subscript) -> Any:
        value = self.__execute_node(node.value)
        if isinstance(node.slice, ast.Slice):
            lower = self.__execute_node(node.slice.lower)
            upper = self.__execute_node(node.slice.upper)
            if isinstance(node.ctx, ast.Store):

                def update_slice(data):
                    self.__add_to_execution_log(
                        f'updating {_display(value)}[{lower}:{upper}] = {data}'
                    )
                    value[lower:upper] = data
                    return

                return update_slice
            return value[lower:upper]

        if isinstance(node.slice, ast.Index):
            index = self.__execute_node(node.slice.value)
            if isinstance(node.ctx, ast.Store):

                def update_index(data):
                    self.__add_to_execution_log(f'updating {_display(value)}[{index}] = {data}')
                    value[index] = data
                    return

                return update_index
            return value[index]
        raise NotImplementedError(f'{_display(node.slice)} subscript slice not supported')

    def __execute_augassign(self, node: ast.AugAssign) -> Any:
        target_id = self.__execute_node(node.target)
        return self.__execute_node(
            ast.Assign(
                targets=[node.target],
                value=ast.BinOp(
                    left=ast.Name(id=target_id, ctx=ast.Load()), op=node.op, right=node.value
                ),
            )
        )

    def __store(self, target: Union[Callable, str, Tuple], value) -> None:
        if callable(target):
            target(value)
            return
        if isinstance(target, str):
            self.__add_to_execution_log(f'setting {target} = {value}')
            self.mem.update({target: value})
            return
        if isinstance(target, Iterable) and isinstance(value, Iterable):
            for t, v in zip(target, value):
                self.__store(t, v)
            return
        raise SyntaxError(f'{_display(target)} not valid type for assignment')

    def __execute_assign(self, node: ast.Assign) -> None:
        value = self.__execute_node(node.value)

        for target_node in node.targets:
            if isinstance(target_node, ast.Tuple) or isinstance(target_node, ast.List):
                for unpacked_node, unpacked_value in zip(target_node.elts, value):
                    unpacked_target = self.__execute_node(unpacked_node)
                    self.__store(unpacked_target, unpacked_value)
            else:
                target = self.__execute_node(target_node)
                self.__store(target, value)
        return None

    def __execute_if(self, node: ast.If) -> None:
        test = self.__execute_node(node.test)
        if test:
            self.__execute_node_list(node.body)
        else:
            self.__execute_node_list(node.orelse)
        return None

    def __execute_for(self, node: ast.For) -> None:
        target = self.__execute_node(node.target)
        iterable = self.__execute_node(node.iter)
        self.__add_to_execution_log(f'iterating through \'{node.target.id}\'')

        broken = False
        for el in iterable:
            self.__store(target, el)
            for body_node in node.body:
                try:
                    self.__execute_node(body_node)
                except Break:
                    broken = True
                    break
                except Continue:
                    break
            if broken:
                break

        if not broken:
            self.__execute_node_list(node.orelse)
        return None

    def __execute_while(self, node: ast.While) -> None:
        test = self.__execute_node(node.test)
        broken = False
        while test:
            try:
                self.__execute_node_list(node.body)
                test = self.__execute_node(node.test)
            except Break:
                broken = True
                break
            except Continue:
                continue

        if not broken:
            self.__execute_node_list(node.orelse)
        return None

    def __execute_break(self, node: ast.Break) -> ast.Break:
        raise Break()

    def __execute_continue(self, node: ast.Continue) -> ast.Continue:
        raise Continue()

    def __execute_return(self, node: ast.Return) -> None:
        value = self.__execute_node(node.value)
        raise Return(value, 'program return')

    def __execute_starred(self, node: ast.Starred) -> Any:
        return self.__execute_node(node.value)

    def __execute_call(self, node: ast.Call) -> Any:
        fn = self.__execute_node(node.func)
        args = []
        for arg in node.args:
            val = self.__execute_node(arg)
            if isinstance(arg, ast.Starred):
                args.extend(val)
            else:
                args.append(val)

        kwargs = {}
        for kw in node.keywords:
            key = kw.arg
            value = self.__execute_node(kw.value)
            if key is None:
                kwargs.update(value)
            else:
                kwargs[key] = value
        self.__add_to_execution_log(f'calling {fn.__name__}(*{args}, **{kwargs})')
        return fn(*args, **kwargs)

    def __execute_delete(self, node) -> Any:
        targets = node.targets
        return self.__execute_node_list(targets)

    def __execute_comp(
        self,
        node: Union[ast.ListComp, ast.SetComp, ast.DictComp],
    ) -> Union[List, Set, Dict]:
        comp_target = FN_COMPREHENSION_INIT[type(node)]()

        num_gen = len(node.generators)
        if num_gen == 0:
            return comp_target
        elif num_gen == 1:
            fn = FN_COMPREHENSION_SINGLE[type(node)]
        else:
            fn = FN_COMPREHENSION_MULTI[type(node)]

        self.mem.push()
        target = self.__execute_node(node.generators[0].target)
        iterable = self.__execute_node(node.generators[0].iter)
        for el in iterable:
            self.__store(target, el)

            ifs = True
            for bool_node in node.generators[0].ifs:
                val = self.__execute_node(bool_node)
                ifs = ifs and val

            if not ifs:
                continue

            fn(self.__execute_node, comp_target, node)

        self.mem.pop()
        return comp_target

    def __execute_listcomp(self, node: ast.ListComp) -> List:
        return self.__execute_comp(node)

    def __execute_setcomp(self, node: ast.SetComp) -> List:
        return self.__execute_comp(node)

    def __execute_dictcomp(self, node: ast.DictComp) -> List:
        return self.__execute_comp(node)

    def __execute_generator_exp(self, node: ast.GeneratorExp) -> Generator:
        local_mem = {}
        num_gen = len(node.generators)
        if num_gen == 0:
            return

        target = self.__execute_node(node.generators[0].target)
        iterable = self.__execute_node(node.generators[0].iter)

        for el in iterable:
            self.mem.push(local_mem)
            self.__store(target, el)

            ifs = True
            for bool_node in node.generators[0].ifs:
                val = self.__execute_node(bool_node)
                ifs = ifs and val

            if not ifs:
                continue

            if num_gen > 1:
                nested_generator = self.__execute_node(
                    ast.GeneratorExp(elt=node.elt, generators=node.generators[1:])
                )
                while True:
                    try:
                        next_value = next(nested_generator)
                        local_mem = self.mem.pop()
                        yield next_value
                        self.mem.push(local_mem)
                    except StopIteration:
                        self.mem.pop()
                        break
            else:
                next_value = self.__execute_node(node.elt)
                self.mem.pop()
                yield next_value

    def __execute_lambda(self, node: ast.Lambda) -> Callable:
        node_args = [arg.arg for arg in node.args.args]
        defaults = []
        for default_node in node.args.defaults:
            val = self.__execute_node(default_node)
            defaults.append(val)

        def lambda_fn(*args, **kwargs):
            if len(args) + len(kwargs) > len(node_args):
                raise TypeError(
                    f'<lambda>() takes {len(node_args)} args but {len(args)} were given'
                )
            mem_index = len(self.mem)
            self.mem.push()
            for i, arg in enumerate(node_args):
                if i < len(args):
                    val = args[i]
                elif arg in kwargs:
                    val = kwargs[arg]
                else:
                    val = defaults[i - (len(node_args) - len(defaults))]

                self.__store(arg, val)

            out = self.__execute_node(node.body)
            self.mem.pop(mem_index)
            return out

        return lambda_fn

    def __execute_function_def(self, node: ast.FunctionDef):
        node_args = [arg.arg for arg in node.args.args]
        defaults = []
        for default_node in node.args.defaults:
            val = self.__execute_node(default_node)
            defaults.append(val)

        def fn(*args, **kwargs):
            if len(args) + len(kwargs) > len(node_args):
                raise TypeError(
                    f'<{node.name}>() takes {len(node_args)} args but {len(args)} were given'
                )
            mem_index = len(self.mem)
            self.mem.push()
            for i, arg in enumerate(node_args):
                if i < len(args):
                    val = args[i]
                elif arg in kwargs:
                    val = kwargs[arg]
                else:
                    val = defaults[i - (len(node_args) - len(defaults))]

                self.__store(arg, val)
            try:
                self.__execute_node_list(node.body)
                self.mem.pop(mem_index)
                return None
            except Return as e:
                self.mem.pop(mem_index)
                return e.value

        self.__store(node.name, fn)
        return fn

    def __execute_formatted_value(self, node: ast.FormattedValue):
        return str(self.__execute_node(node.value))

    def __execute_joinedstr(self, node: ast.JoinedStr) -> str:
        if len(node.values) == 0:
            return ''

        return self.__execute_node(node.values[0]) + self.__execute_node(
            ast.JoinedStr(values=node.values[1:])
        )


class SizeValidator:
    def __init__(
        self,
        supported_primitives_types: Set,
        max_validation_size: int = 8192,
        max_string_length: int = 4096,
        max_iterable_length: int = 2048,
        max_constant: int = 4294967296,
        min_constant: int = -4294967296,
    ):
        self._supported_primitives_types = supported_primitives_types
        self._max_validation_size = max_validation_size
        self._max_string_length = max_string_length
        self._max_iterable_length = max_iterable_length
        self._max_constant = max_constant
        self._min_constant = min_constant

        self._type_map = {
            type(None): self._default_size,
            int: self._numerical_primitive_size,
            float: self._numerical_primitive_size,
            dict: self._dict_size,
            str: self._string_size,
            list: self._iterable_size,
            set: self._iterable_size,
            tuple: self._iterable_size,
            GeneratorType: self._default_size,
            BuiltinFunctionType: self._default_size,
            BuiltinMethodType: self._default_size,
            LambdaType: self._default_size,
            FunctionType: self._default_size,
            bool: self._default_size,
            range: self._default_size,
            reversed: self._default_size,
            MethodType: self._default_size,
            type: self._default_size,
        }
        for primitive_type in supported_primitives_types:
            if primitive_type not in self._type_map:
                self._type_map[primitive_type] = self._default_size

    def check_size(self, value, size: int):
        if size > self._max_validation_size:
            raise MaximumSizeExceededError(
                f'{size} of {_display(value)} exceeds maximum memory constraints length: {self._max_validation_size}'
            )

    def _dict_size(self, value: dict, checked: Set) -> int:
        if id(value) in checked:
            return 1

        size = 0
        checked.add(id(value))
        if len(value) > self._max_iterable_length:
            raise MaximumSizeExceededError(
                f'dict of {len(value)} elements exceeds maximum allowed: {self._max_iterable_length}'
            )
        for k, v in value.items():
            size += self.get_size(k, checked)
            size += self.get_size(v, checked)
        return size

    def _numerical_primitive_size(self, value: Union[int, float], _: Set) -> int:
        if isinstance(value, (int, float)) and not (
            self._min_constant <= value <= self._max_constant
        ):
            raise MaximumSizeExceededError(
                f'{value} not in range {[self._min_constant, self._max_constant]}'
            )
        return 1

    def _string_size(self, value: str, _: Set) -> int:
        if len(value) > self._max_string_length:
            raise MaximumSizeExceededError(
                f'str \'{value}\' exceeds maximum length of {self._max_string_length}'
            )
        return 1

    def _iterable_size(self, value: Iterable, checked: Set) -> int:
        if id(value) in checked:
            return 1

        checked.add(id(value))
        size = 0
        if hasattr(value, '__len__') and len(value) > self._max_iterable_length:
            raise MaximumSizeExceededError(
                f'iterable of {len(value)} elements exceeds maximum allowed: {self._max_iterable_length}'
            )
        iterable = iter(value)
        # We make the assumption that iter returns a new iterator that isn't affected by next()
        for el in iterable:
            size += self.get_size(el, checked)
            self.check_size(value, size)
        return size

    def _default_size(self, *args) -> int:
        return 1

    def get_size(self, value: Any, checked: Set):
        if type(value) in self._type_map:
            return self._type_map[type(value)](value, checked)
        if isinstance(value, Iterable):
            return self._iterable_size(value, checked)
        raise NotImplementedError(f'type of {value} not supported')

    def validate_size(self, value: Any, checked: Set) -> None:
        self.check_size(value, self.get_size(value, checked))

    def validate_memory(self, memory: MemoryStack) -> None:
        self.validate_size(memory, set())
