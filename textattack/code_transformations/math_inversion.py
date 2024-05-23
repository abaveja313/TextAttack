import ast
from typing import Type

from textattack.code_transformations.mutation import (
    OneByOneVisitor,
    OneByOneTransformer,
)


class MathInversionVisitor(OneByOneVisitor):
    @property
    def name(self):
        return "MathInversion"

    def is_transformable(self, node):
        return isinstance(node, ast.BinOp) and (
            isinstance(node.op, ast.Add)
            or isinstance(node.op, ast.Sub)
            or isinstance(node.op, ast.Div)
            or isinstance(node.op, ast.FloorDiv)
            or isinstance(node.op, ast.Mod)
            or isinstance(node.op, ast.Pow)
        )

    def transform_node(self, node):
        if isinstance(node.op, ast.Add):
            node.op = ast.Sub()
            node.right = ast.UnaryOp(op=ast.USub(), operand=node.right)
        elif isinstance(node.op, ast.Sub):
            node.op = ast.Add()
            node.right = ast.UnaryOp(op=ast.USub(), operand=node.right)
        elif isinstance(node.op, ast.Mult):
            node.op = ast.Div()
            node.right = ast.BinOp(
                op=ast.Div(), left=ast.Constant(1.0), right=node.right
            )
        elif isinstance(node.op, ast.Div):
            node.op = ast.Mult()
            node.right = ast.BinOp(
                op=ast.Mult(), left=ast.Constant(1.0), right=node.right
            )
            return ast.Call(
                func=ast.Name(id="int", ctx=ast.Load()),
                args=[
                    ast.BinOp(
                        left=node.left,
                        op=ast.Div(),
                        right=ast.BinOp(
                            left=ast.Constant(1.0), op=ast.Div(), right=node.right
                        ),
                    )
                ],
                keywords=[],
            )
        elif isinstance(node.op, ast.Mod):
            node.op = ast.Sub()
            node.right = ast.BinOp(
                op=ast.Mult(),
                left=node.right,
                right=ast.BinOp(op=ast.FloorDiv(), left=node.left, right=node.right),
            )

        return node


class MathInversionTransformer(OneByOneTransformer):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return MathInversionVisitor
