import ast
from typing import Type

from textattack.code_transformations.mutation import OneByOneVisitor, OneByOneTransformer


class IfToWhileLoopVisitor(OneByOneVisitor):
    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        while_loop = ast.While(
            test=node.test,
            body=[ast.Pass()]
        )
        return while_loop

    def is_transformable(self, node):
        return isinstance(node, ast.If)


class IfToWhileLoopTransformer(OneByOneTransformer):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return IfToWhileLoopVisitor
