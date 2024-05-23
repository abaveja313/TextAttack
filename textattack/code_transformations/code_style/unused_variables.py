import ast
from random import random
from typing import Type

from textattack.code_transformations import CRT
from textattack.code_transformations.mutation import (
    OneByOneTransformer,
    OneByOneVisitor,
)


class UnusedVariableVisitor(OneByOneVisitor, categories=CRT.semantics):

    @staticmethod
    def weird_assign():
        return ast.Assign(
            targets=[ast.Name('foo')],
            value=random.randint(1, 10)
        )

    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        return ast.Module(body=[node, self.weird_assign()])

    def is_transformable(self, node):
        return isinstance(node, ast.stmt)


class UnusedVariableTransformer(OneByOneTransformer):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return UnusedVariableVisitor
