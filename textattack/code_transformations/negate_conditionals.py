import ast
from typing import Type

from textattack.code_transformations.mutation import (
    OneByOneVisitor,
    OneByOneTransformer,
)


class NegateConditionalVisitor(OneByOneVisitor):
    @property
    def name(self):
        return "NegateConditionalVisitor"

    def is_transformable(self, node):
        return isinstance(node, ast.If)

    def transform_node(self, node):
        conditions = [node.test]

        def collect_conditions(orelse):
            for elif_node in orelse:
                if isinstance(elif_node, ast.If):
                    conditions.append(elif_node.test)
                    collect_conditions(elif_node.orelse)
                else:
                    break

        collect_conditions(node.orelse)

        negated_condition = ast.UnaryOp(
            op=ast.Not(), operand=ast.BoolOp(op=ast.Or(), values=conditions)
        )

        pass_node = ast.Pass()

        return ast.If(test=negated_condition, body=[pass_node], orelse=[])


class NegateConditionalTransformer(OneByOneTransformer):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return NegateConditionalVisitor
