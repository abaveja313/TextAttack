import ast
import copy
import warnings

from textattack.shared import AttackedText
from textattack.transformations import Transformation

import ast
import copy


class BooleanTransformer(ast.NodeTransformer):
    def __init__(self, code):
        self.code = code
        self.transformations = []
        self.original_tree = ast.parse(self.code)

    def transform(self):
        for node in ast.walk(self.original_tree):
            if isinstance(node, (ast.If, ast.While, ast.Assign)):
                for target in [node.test] if hasattr(node, 'test') else [node.value]:
                    self.apply_transformations(node, target)
        return [ast.unparse(tree) for tree in self.transformations]

    def apply_transformations(self, node, target):
        transformed = self.transform_expression(copy.deepcopy(target))
        for expr in transformed:
            new_tree = copy.deepcopy(self.original_tree)
            for new_node in ast.walk(new_tree):
                if isinstance(new_node, node.__class__) and ast.dump(new_node) == ast.dump(node):
                    if hasattr(new_node, 'test'):
                        new_node.test = expr
                    else:
                        new_node.value = expr
                    break
            self.transformations.append(new_tree)

    def transform_expression(self, expr):
        results = []
        if isinstance(expr, ast.Compare):
            inverted_expr = self.apply_inversion(expr)
            negated_expr = self.apply_simple_negation(inverted_expr)
            results.append(negated_expr)
        elif isinstance(expr, ast.BoolOp):
            demorgan_expr = self.apply_de_morgans_law(expr)
            results.append(demorgan_expr)
        elif isinstance(expr, ast.UnaryOp) and isinstance(expr.operand, (ast.BoolOp, ast.Compare)):
            simplified_expr = self.simplify_negation(expr)
            results.append(simplified_expr)
        return results

    def apply_simple_negation(self, expr):
        return ast.UnaryOp(op=ast.Not(), operand=expr)

    def apply_de_morgans_law(self, expr):
        if isinstance(expr.op, ast.And):
            new_op = ast.Or()
        else:
            new_op = ast.And()
        new_values = [ast.UnaryOp(op=ast.Not(), operand=v) for v in expr.values]
        return ast.UnaryOp(op=ast.Not(), operand=ast.BoolOp(op=new_op, values=new_values))

    def simplify_negation(self, expr):
        if isinstance(expr.operand, ast.BoolOp):
            if isinstance(expr.operand.op, ast.And):
                new_op = ast.Or()
            else:
                new_op = ast.And()
            new_values = [self.invert(v) for v in expr.operand.values]
            return ast.BoolOp(op=new_op, values=new_values)
        elif isinstance(expr.operand, ast.Compare):
            return self.invert(expr.operand)

    def apply_inversion(self, expr):
        if isinstance(expr, ast.Compare):
            new_ops = []
            for op in expr.ops:
                if isinstance(op, ast.Is):
                    new_ops.append(ast.IsNot())
                if isinstance(op, ast.IsNot):
                    new_ops.append(ast.Is())
                elif isinstance(op, ast.Eq):
                    new_ops.append(ast.NotEq())
                elif isinstance(op, ast.NotEq):
                    new_ops.append(ast.Eq())
                elif isinstance(op, ast.Lt):
                    new_ops.append(ast.GtE())
                elif isinstance(op, ast.Gt):
                    new_ops.append(ast.LtE())
                elif isinstance(op, ast.LtE):
                    new_ops.append(ast.Gt())
                elif isinstance(op, ast.GtE):
                    new_ops.append(ast.Lt())
            return ast.Compare(left=expr.left, ops=new_ops, comparators=expr.comparators)

    def invert(self, expr):
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
            # Correctly handle 'a is False' to 'not (a is not False)'
            if isinstance(expr.operand, ast.Compare) and isinstance(expr.operand.ops[0], ast.Is):
                return ast.UnaryOp(op=ast.Not(), operand=ast.Compare(
                    left=expr.operand.left,
                    ops=[ast.IsNot()],
                    comparators=expr.operand.comparators
                ))
            else:
                return expr.operand
        else:
            return self.apply_inversion(expr)


class IfStatementNegatingTransformation(Transformation):
    def _get_transformations(self, attacked_text, indices_to_modify):
        print("Applying if negation transformation")
        current_text = attacked_text.text
        transformer_nested = BooleanTransformer(current_text)
        refactorings = transformer_nested.transform()
        return [AttackedText(r) for r in refactorings]
