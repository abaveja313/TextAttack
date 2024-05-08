"""

Seq2Sick
================================================
(Seq2Sick: Evaluating the Robustness of Sequence-to-Sequence Models with Adversarial Examples)
"""

from textattack import Attack
from textattack.attack_recipes import AttackRecipe
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.goal_functions import TestcaseFailOutputGoalFunction
from textattack.search_methods import BeamSearch
from textattack.transformations import VariableSwitchingTransformation, IfStatementNegatingTransformation, \
    CompositeTransformation, MathInversionTransformation


class AmritAdversarialAttack(AttackRecipe):
    """Cheng, Minhao, et al.

    Seq2Sick: Evaluating the Robustness of Sequence-to-Sequence Models with
    Adversarial Examples

    https://arxiv.org/abs/1803.01128

    This is a greedy re-implementation of the seq2sick attack method. It does
    not use gradient descent.
    """

    @staticmethod
    def build(model_wrapper, goal_function="non_overlapping"):
        #
        # Goal is non-overlapping output.
        #
        goal_function = TestcaseFailOutputGoalFunction(
            model_wrapper,
            query_budget=20
        )
        transformation = CompositeTransformation(
            transformations=[
                VariableSwitchingTransformation(),
                IfStatementNegatingTransformation(),
                MathInversionTransformation()
            ]
        )
        # transformation = WordSwapEmbedding()

        #
        # Don't modify the same word twice or stopwords
        #
        constraints = [RepeatModification(), StopwordModification()]
        #
        # In these experiments, we hold the maximum difference
        # on edit distance (ϵ) to a constant 30 for each sample.
        #
        #
        # Greedily swap words with "Word Importance Ranking".
        #
        search_method = BeamSearch()

        return Attack(goal_function, constraints, transformation, search_method)
