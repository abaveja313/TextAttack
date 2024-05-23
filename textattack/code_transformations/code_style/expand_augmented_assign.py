import ast
from typing import Type

from textattack.code_transformations.mutation import (
    OneByOneTransformer,
    OneByOneVisitor,
)


class ExpandAugmentedAssignVisitor(OneByOneVisitor):
    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        new_value = ast.BinOp(left=node.target, op=node.op, right=node.value)
        return ast.Assign(targets=[node.target], value=new_value)

    def is_transformable(self, node):
        return isinstance(node, ast.AugAssign)


class ExpandAugmentedAssignTransformer(OneByOneTransformer):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return ExpandAugmentedAssignVisitor
