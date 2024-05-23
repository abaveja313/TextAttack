import ast
from typing import Type

from textattack.code_transformations.mutation import OneByOneVisitor, OneByOneTransformer


class EmptyArrayToStringVisitor(OneByOneVisitor):
    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        node.value = ast.Constant(value="")
        return node

    def is_transformable(self, node):
        return isinstance(node, ast.Assign) and isinstance(node.value, ast.List) and len(node.value.elts) == 0


class EmptyArrayToStringTransformer(OneByOneTransformer):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return EmptyArrayToStringVisitor
