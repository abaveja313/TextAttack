import ast
from typing import List


class ASTTokenizer(ast.NodeVisitor):
    def __init__(self, code: str):
        self.code = code
        self.ast_tree = ast.parse(self.code)
        self.tokens = []
        self.path = []

        self.index = 0

    def generic_visit(self, node):
        identifier = f"{type(node).__name__}-{self.index}"
        self.tokens.append("_".join([*self.path, identifier]))
        self.path.append(identifier)
        self.index += 1
        super().generic_visit(node)
        self.path.pop()

    @staticmethod
    def dfs_walk(node):
        yield node
        for child in ast.iter_child_nodes(node):
            yield from ASTTokenizer.dfs_walk(child)

    @staticmethod
    def tokenize(code: str) -> List[str]:
        tokenizer = ASTTokenizer(code)
        for node in ASTTokenizer.dfs_walk(tokenizer.ast_tree):
            tokenizer.visit(node)
        return tokenizer.tokens

    def visit_Store(self, node):
        pass

    def visit_Load(self, node):
        pass


test_code = """
for i in range(10):
    if i == 10:
        print(i)
"""

tree = ASTTokenizer.tokenize(test_code)
print(tree)
