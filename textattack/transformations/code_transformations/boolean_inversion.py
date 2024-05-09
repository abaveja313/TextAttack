import ast
from typing import Type

from textattack.transformations.code_transformations.mutation import OneByOneVisitor, OneByOneTransformer


class BooleanInversionVisitor(OneByOneVisitor):
    @property
    def name(self):
        return "BooleanTransformer"

    def is_transformable(self, node):
        return (isinstance(node, ast.If) or
                isinstance(node, ast.While) or
                (isinstance(node, ast.Assign) and self.is_boolean_assignment(node)))

    def transform_expression(self, node: ast.expr):
        if isinstance(node, ast.Compare):
            inverted_expr = self.apply_inversion(node)
            negated_expr = self.apply_simple_negation(inverted_expr)
            return negated_expr
        elif isinstance(node, ast.BoolOp):
            demorgan_expr = self.apply_de_morgans_law(node)
            return demorgan_expr
        elif isinstance(node, ast.UnaryOp) and isinstance(node.operand, (ast.BoolOp, ast.Compare)):
            simplified_expr = self.simplify_negation(node)
            return simplified_expr
        else:
            # Handle generic truthy/falsy expressions
            double_negation = self.apply_double_negation(node)
            return double_negation

    def transform_node(self, node: ast.AST):
        if isinstance(node, (ast.If, ast.While)):
            node.test = self.transform_expression(node.test)
        elif isinstance(node, ast.Assign):
            node.value = self.transform_expression(node.value)
        return [node]

    def is_boolean_assignment(self, node):
        if isinstance(node.value, ast.UnaryOp) and isinstance(node.value.op, ast.Not):
            return True
        elif isinstance(node.value, ast.Compare):
            return True
        return False

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
                elif isinstance(op, ast.IsNot):
                    new_ops.append(ast.Is())
                elif isinstance(op, ast.In):
                    new_ops.append(ast.NotIn())
                elif isinstance(op, ast.NotIn):
                    new_ops.append(ast.In())
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

    def apply_double_negation(self, expr):
        # Apply double negation for truthy/falsy expressions
        return ast.UnaryOp(op=ast.Not(), operand=ast.UnaryOp(op=ast.Not(), operand=expr))

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


class BooleanInversionTransformer(OneByOneTransformer):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return BooleanInversionVisitor
