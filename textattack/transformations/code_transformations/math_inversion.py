import ast
import copy

from textattack.shared import AttackedText
from textattack.shared.utils import parse_stem
from textattack.transformations import Transformation


class Transformer(ast.NodeTransformer):
    def __init__(self, transformation):
        self.transformation = transformation
        self.transformation_applied = False

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Add) and self.transformation == 'add_sub':
            node.op = ast.Sub()
            node.right = ast.UnaryOp(op=ast.USub(), operand=node.right)
            self.transformation_applied = True
        elif isinstance(node.op, ast.Sub) and self.transformation == 'sub_add':
            node.op = ast.Add()
            node.right = ast.UnaryOp(op=ast.USub(), operand=node.right)
            self.transformation_applied = True
        elif isinstance(node.op, ast.Mult) and self.transformation == 'mult_div':
            node.op = ast.Div()
            node.right = ast.Call(
                func=ast.Name(id='pow', ctx=ast.Load()),
                args=[node.right, ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=1))],
                keywords=[]
            )
            self.transformation_applied = True
        elif isinstance(node.op, ast.Div) and self.transformation == 'div_mult':
            node.op = ast.Mult()
            node.right = ast.Call(
                func=ast.Name(id='pow', ctx=ast.Load()),
                args=[node.right, ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=1))],
                keywords=[]
            )
            self.transformation_applied = True
        return node

    def visit_AugAssign(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Add) and self.transformation == 'aug_add_sub':
            node.op = ast.Sub()
            node.value = ast.UnaryOp(op=ast.USub(), operand=node.value)
            self.transformation_applied = True
        elif isinstance(node.op, ast.Sub) and self.transformation == 'aug_sub_add':
            node.op = ast.Add()
            node.value = ast.UnaryOp(op=ast.USub(), operand=node.value)
            self.transformation_applied = True
        elif isinstance(node.op, ast.Mult) and self.transformation == 'aug_mult_div':
            node.op = ast.Div()
            node.value = ast.Call(
                func=ast.Name(id='pow', ctx=ast.Load()),
                args=[node.value, ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=1))],
                keywords=[]
            )
            self.transformation_applied = True
        elif isinstance(node.op, ast.Div) and self.transformation == 'aug_div_mult':
            node.op = ast.Mult()
            node.value = ast.Call(
                func=ast.Name(id='pow', ctx=ast.Load()),
                args=[node.value, ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=1))],
                keywords=[]
            )
            self.transformation_applied = True
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == 'pow' and self.transformation == 'pow_exp':
            new_node = ast.BinOp(
                left=node.args[0],
                op=ast.Pow(),
                right=node.args[1]
            )
            self.transformation_applied = True
            return new_node
        return node


class MathInversionTransformation(Transformation):
    def _get_transformations(self, attack_text, indices_to_modify):
        print("Applying math inversion transformation")
        current_text = attack_text.text
        variations = self.refactor_code(current_text)
        stems = [parse_stem(current_text, r) for r in variations]
        return [AttackedText(variation) for variation in stems]

    @staticmethod
    def refactor_code(code):
        # Parse the code into an AST
        tree = ast.parse(code)
        # List to store different refactored versions of the code
        refactored_versions = []
        # Define different transformers for each rule
        transformations = [
            'add_sub', 'sub_add', 'mult_div', 'div_mult',
            'aug_add_sub', 'aug_sub_add', 'aug_mult_div', 'aug_div_mult', 'pow_exp'
        ]

        # Apply each transformation and gather the results
        for transformation in transformations:
            temp_tree = copy.deepcopy(tree)
            transformer = Transformer(transformation)
            transformer.visit(temp_tree)
            if transformer.transformation_applied:
                refactored_versions.append(ast.unparse(temp_tree))

        return refactored_versions
