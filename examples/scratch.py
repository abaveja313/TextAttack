import sys

from evalplus.data import get_human_eval_plus

data = get_human_eval_plus()
example = data['HumanEval/112']['prompt'] + data['HumanEval/112']['canonical_solution']

from textattack.llms.evalplus_wrapper import EvalPlusWrapper
from textattack.llms.models import GeneralVllmDecoder
from textattack.attack_recipes.amrit_adversarial import AmritAdversarialAttack
from textattack.truth_initializers import TestPassingInitializer
from loguru import logger

logger.add(sys.stdout, format="{time} {file} {level} {message}")
ground_truth_output = "HumanEval/112"

model = GeneralVllmDecoder(
    'deepseek-ai/deepseek-coder-1.3b-instruct',
    dataset='humaneval',
    tp=1,
    temperature=0.95,
    dtype="half"
)

wrapper = EvalPlusWrapper(model)

initializer = TestPassingInitializer(
    model=wrapper,
    task_id=ground_truth_output,
    num_samples=200,
    batch_size=100
)

initial_solution = initializer.find_centroid_solution()
logger.info(f"Solution:\n{initial_solution}")

recipe = AmritAdversarialAttack.build(wrapper)

print(initial_solution)
a = recipe.attack(initial_solution, ground_truth_output)
print(a)
