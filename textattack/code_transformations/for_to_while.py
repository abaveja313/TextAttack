import ast
from typing import Type

from textattack.code_transformations.mutation import (
    OneByOneVisitor,
    OneByOneTransformer,
)


class ForToWhileVisitor(OneByOneVisitor):
    @property
    def name(self):
        return "ForToWhile"

    def is_transformable(self, node):
        return (
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
        )

    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        range_args = node.iter.args

        if len(range_args) == 1:  # range(stop)
            start = ast.Constant(0, lineno=node.lineno, col_offset=node.col_offset)
            stop = range_args[0]
        elif len(range_args) == 2 or len(range_args) == 3:  # range(start, stop, [step])
            start = range_args[0]
            stop = range_args[1]
        else:
            raise ValueError("Invalid number of arguments to `range`")

        index_var = node.target.id
        init_assign = ast.Assign(
            targets=[
                ast.Name(
                    id=index_var,
                    ctx=ast.Store(),
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                )
            ],
            value=start,
            lineno=node.lineno,
            col_offset=node.col_offset,
        )

        condition = ast.Compare(
            left=ast.Name(
                id=index_var,
                ctx=ast.Load(),
                lineno=node.lineno,
                col_offset=node.col_offset,
            ),
            ops=[ast.Lt()],
            comparators=[stop],
            lineno=node.lineno,
            col_offset=node.col_offset,
        )

        while_body = [ast.Pass(lineno=node.lineno, col_offset=node.col_offset)]
        new_while = ast.While(
            test=condition,
            body=while_body,
            orelse=[],
            lineno=node.lineno,
            col_offset=node.col_offset,
        )

        return ast.Module(body=[init_assign, new_while], type_ignores=[])


class ForToWhileTransformer(OneByOneTransformer):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return ForToWhileVisitor
