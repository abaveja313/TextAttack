# We want to implement
# Base 10 -> Base 16
# Base 10 -> Base 8
# Base 10 -> Base 2
import ast
from abc import abstractmethod, ABC
from typing import Type

from textattack.code_transformations import CRT
from textattack.code_transformations.mutation import OneByOneTransformer, OneByOneVisitor


class DecimalBaseVisitor(OneByOneVisitor, ABC):
    @abstractmethod
    def convert(self, num: int) -> str:
        pass

    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        if isinstance(node, ast.Constant):
            new_node = ast.Constant(
                value=self.convert(node.value),
                kind=None
            )
            return new_node
        return node

    def is_transformable(self, node):
        # does this work with Py3.8 num
        return isinstance(node, ast.Constant) and isinstance(node.value, int)


class DecimalBase2Visitor(DecimalBaseVisitor):
    def convert(self, num: int) -> str:
        return bin(num)


class DecimalBase8Visitor(DecimalBaseVisitor):
    def convert(self, num: int) -> str:
        return oct(num)


class DecimalBase16Visitor(DecimalBaseVisitor):
    def convert(self, num: int) -> str:
        return hex(num)


class DecimalBase2Transformer(OneByOneTransformer, categories=CRT.lexical):

    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return DecimalBase2Visitor


class DecimalBase8Transformer(OneByOneTransformer, categories=CRT.lexical):

    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return DecimalBase8Visitor


class DecimalBase16Transformer(OneByOneTransformer, categories=CRT.lexical):

    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return DecimalBase16Visitor
