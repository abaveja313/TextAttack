import ast
import difflib
from typing import List

import tqdm


def ast_dump(code: str) -> str:
    return ast.dump(ast.parse(code))


def calculate_similarity(ast1: str, ast2: str) -> float:
    sequence_matcher = difflib.SequenceMatcher(None, ast1, ast2)
    return sequence_matcher.ratio()


def find_centroid_program(codes: List[str]) -> str:
    asts = [ast_dump(code) for code in codes]
    n = len(asts)
    similarity_matrix = [[0] * n for _ in range(n)]

    for i in tqdm.tqdm(range(n), desc="Calculating Similarity"):
        for j in range(i + 1, n):
            similarity = calculate_similarity(asts[i], asts[j])
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity  # Symmetric matrix

    # Calculate the average similarity for each program
    average_similarities = [
        sum(similarities) / (n - 1) for similarities in similarity_matrix
    ]

    # Find the index of the program with the maximum average similarity
    centroid_index = max(range(n), key=lambda i: average_similarities[i])
    return codes[centroid_index]
