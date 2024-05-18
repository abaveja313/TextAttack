import ast
import random
import string
from typing import Type

from textattack.transformations.code_transformations.mutation import (
    OneByOneTransformer,
    OneByOneVisitor,
)


class PrintInjectionVisitor(OneByOneVisitor):
    @property
    def name(self):
        return "PrintInjection"

    @staticmethod
    def debug_statement():
        return ast.Call(
            func=ast.Name(id="print", ctx=ast.Load()),
            args=[ast.Constant("This line was reached for debugging!")],
        )

    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        return ast.Module(body=[node, self.debug_statement()])

    def is_transformable(self, node):
        return isinstance(node, ast.Expr) or isinstance(node, ast.Assign)


class PrintInjectionTransformer(OneByOneTransformer):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return PrintInjectionVisitor
