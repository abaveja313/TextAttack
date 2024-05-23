import ast
from typing import Type

from textattack.code_transformations.mutation import OneByOneTransformer, OneByOneVisitor


class MultiplyBy2ToBitshiftVisitor(OneByOneVisitor):
    def is_transformable(self, node):
        return isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mul) and \
            isinstance(node.right, ast.Constant) and node.right.value == 2

    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        node.op = ast.LShift
        node.right = ast.Constant(n=1)
        return node


class DivideBy2ToBitshiftVisitor(OneByOneVisitor):
    def is_transformable(self, node):
        return isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and \
            isinstance(node.right, ast.Constant) and node.right.value == 2

    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        node.op = ast.RShift
        node.right = ast.Constant(n=1)
        return node


class NegationToComplementVisitor(OneByOneVisitor):
    def is_transformable(self, node):
        return isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)

    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        new_node = ast.BinOp(
            op=ast.Add(),
            left=ast.UnaryOp(op=ast.Invert(), operand=node.operand),
            right=1
        )
        return new_node


class MultiplyBy2ToBitshiftTransformer(OneByOneTransformer):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return MultiplyBy2ToBitshiftVisitor


class DivideBy2ToBitshiftTransformer(OneByOneTransformer):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return DivideBy2ToBitshiftVisitor


class NegationToComplementTransformer(OneByOneTransformer):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return NegationToComplementVisitor
