import ast
import random
import string
from typing import Type

from textattack.transformations.code_transformations.mutation import (
    OneByOneTransformer,
    OneByOneVisitor,
)


class VariableRenamingVisitor(OneByOneVisitor):
    @property
    def name(self):
        return "VariableRenamer"

    @staticmethod
    def random_variable_name(k=8):
        return ast.Name("".join(random.choices(string.ascii_letters, k=k)))

    def is_transformable(self, node):
        return (
            (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name))
            or (isinstance(node, ast.For) and isinstance(node.target, ast.Name))
            or (
                isinstance(node, ast.With)
                and node.items
                and isinstance(node.items[0].optional_vars, ast.Name)
            )
        )

    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        if isinstance(node, ast.Assign):
            for i in range(len(node.targets)):
                if isinstance(node.targets[i], ast.Name):
                    node.targets[i] = self.random_variable_name()
        elif isinstance(node, ast.For):
            node.target = self.random_variable_name()
        elif isinstance(node, ast.With):
            node.items[0].optional_vars = self.random_variable_name()

        return node


class VariableSwitchingTransformer(OneByOneTransformer):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return VariableRenamingVisitor

    @property
    def deterministic(self):
        return False
