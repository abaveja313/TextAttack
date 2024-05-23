import ast
from abc import ABC, abstractmethod
from typing import Type, Callable, Tuple

from loguru import logger

from textattack.shared.attacked_code import AttackedCode
from textattack.shared.utils import parse_stem
from textattack.tokenization.ast_tokenizer import ASTTokenizer
from textattack.transformations import Transformation
from textattack.code_transformations.registry import RegistedMixin


class ASTCopier(ast.NodeTransformer):
    def generic_visit(self, node: ast.AST):
        if isinstance(node, ast.AST):
            new_node = type(node)()
            for field, value in ast.iter_fields(node):
                setattr(new_node, field, self.visit(value))
            if hasattr(node, "lineno"):
                new_node.lineno = node.lineno
            if hasattr(node, "col_offset"):
                new_node.col_offset = node.col_offset
            if hasattr(node, "end_lineno"):
                new_node.end_lineno = node.end_lineno
            if hasattr(node, "end_col_offset"):
                new_node.end_col_offset = node.end_col_offset
            return new_node
        elif isinstance(node, list):
            return [self.visit(item) for item in node]
        else:
            return node


class NodeReplacer(ast.NodeTransformer):
    def __init__(self, old_node, new_node):
        self.old_node = old_node
        self.new_node = new_node

    def visit(self, node: ast.AST):
        if isinstance(node, type(self.old_node)) and ast.dump(node) == ast.dump(
                self.old_node
        ):
            return self.new_node
        else:
            return self.generic_visit(node)


class OneByOneVisitor(ABC, ASTTokenizer):
    def __init__(self, code: str):
        self.transformations: list[ast.AST] = []
        self.source_code = code
        super().__init__(code)

    @property
    def name(self):
        return self.__class__.__name__

    @abstractmethod
    def is_transformable(self, node):
        pass

    @abstractmethod
    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        pass

    def transform(self) -> list[str]:
        for i, node in enumerate(ASTTokenizer.dfs_walk(self.ast_tree)):
            if i in self.is_transformable(node):
                self.apply_transformations(node)
        return [ast.unparse(tree) for tree in self.transformations]

    def apply_transformations(self, node):
        # Visit a copy of the AST syntax tree
        perturbed_nodes: list[ast.AST] | ast.AST = self.transform_node(
            ASTCopier().visit(node)
        )
        if not isinstance(perturbed_nodes, list):
            perturbed_nodes = [perturbed_nodes]

        for perturbed in perturbed_nodes:
            new_tree = ASTCopier().visit(self.ast_tree)
            perturbed_tree = NodeReplacer(node, perturbed).visit(new_tree)
            self.transformations.append(perturbed_tree)


class PostprocessingTransformer(Transformation, RegistedMixin, ABC):
    @property
    def transformations(self) -> Tuple[Callable[[str, str], str]]:
        return (parse_stem,)

    @property
    @abstractmethod
    def deterministic(self):
        pass

    @property
    @abstractmethod
    def attack_func(self) -> Callable[[str], list[str]]:
        pass

    def postprocess(self, current: str, targets: list[str]) -> list[str]:
        results = []
        for target in targets:
            result = target
            for transformation in self.transformations:
                result = transformation(current, result)
            results.append(result)

        return results

    def _get_transformations(
            self, current_text: AttackedCode, _: list[int]
    ) -> list[AttackedCode]:
        if not isinstance(current_text, AttackedCode):
            raise ValueError("Parameter `current_text` must be of type AttackedCode")

        attacked_text: str = current_text.text
        transformed = self.attack_func(attacked_text)
        post_processed: list[str] = self.postprocess(attacked_text, transformed)

        results = [
            current_text.generate_new_attacked_code(output) for output in post_processed
        ]
        logger.debug(f"{self.__class__.__name__} produced {len(results)} transformations")
        return results


class OneByOneTransformer(PostprocessingTransformer, ABC):
    @property
    def deterministic(self):
        return True

    @property
    @abstractmethod
    def visitor(self) -> Type[OneByOneVisitor]:
        pass

    def _visit_transform(self, source):
        return self.visitor(source).transform()

    @property
    def attack_func(self) -> Callable[[str], list[str]]:
        return self._visit_transform
