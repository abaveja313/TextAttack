import ast
import random
import string
from typing import Generator

from textattack.shared import AttackedText
from textattack.shared.utils import parse_stem
from textattack.transformations import Transformation


class VariableRenamer(ast.NodeTransformer):
    def __init__(self, transformation, target_name=None, new_name=None):
        self.transformation = transformation
        self.target_name = target_name
        self.new_name = new_name or ''.join(random.choices(string.ascii_letters, k=8))

    def visit_Assign(self, node):
        if self.transformation == "assign" and isinstance(node.targets[0], ast.Name):
            target = node.targets[0]
            if target.id == self.target_name:
                target.id = self.new_name
        return self.generic_visit(node)

    def visit_For(self, node):
        if self.transformation == "for_loop" and isinstance(node.target, ast.Name):
            if node.target.id == self.target_name:
                node.target.id = self.new_name
        return self.generic_visit(node)

    def visit_With(self, node):
        if (self.transformation == "with_statement" and node.items
                and isinstance(node.items[0].optional_vars, ast.Name)):
            with_var = node.items[0].optional_vars
            if with_var.id == self.target_name:
                with_var.id = self.new_name
        return self.generic_visit(node)

    def visit_Name(self, node):
        if node.id == self.target_name:
            node.id = self.new_name
        return node


class OneByOneVariableRenamer:
    def __init__(self, source_code):
        self.source_code = source_code
        self.transformations = ["assign", "for_loop", "with_statement"]

    def find_targets(self, tree, transformation):
        targets = []
        for node in ast.walk(tree):
            if (transformation == "assign" and isinstance(node, ast.Assign)
                    and isinstance(node.targets[0], ast.Name)):
                targets.append(node.targets[0].id)
            elif (transformation == "for_loop" and isinstance(node, ast.For)
                  and isinstance(node.target, ast.Name)):
                targets.append(node.target.id)
            elif (transformation == "with_statement" and isinstance(node, ast.With) and
                  node.items and isinstance(node.items[0].optional_vars, ast.Name)):
                targets.append(node.items[0].optional_vars.id)
        return targets

    def refactor_code(self, code, transformation, target_name):
        tree = ast.parse(code)
        new_name = ''.join(random.choices(string.ascii_letters, k=8))
        transformer = VariableRenamer(transformation, target_name, new_name)
        new_tree = transformer.visit(tree)
        return ast.unparse(new_tree)

    def yield_transformations(self) -> Generator[str, None, None]:
        for transformation in self.transformations:
            tree = ast.parse(self.source_code)
            targets = self.find_targets(tree, transformation)
            for target_name in targets:
                refactored_code = self.refactor_code(self.source_code,
                                                     transformation, target_name)
                yield refactored_code


class VariableSwitchingTransformation(Transformation):
    def _get_transformations(self, attacked_text, indices_to_modify):
        current_text = attacked_text.text
        renamer = OneByOneVariableRenamer(current_text)
        refactorings = list(renamer.yield_transformations())

        stems = [parse_stem(current_text, r) for r in refactorings]
        return [AttackedText(r) for r in stems]

    @property
    def deterministic(self):
        return False
