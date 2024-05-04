import functools
import time

from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
from evalplus.evaluate import get_groundtruth, check_correctness

from .text_to_text_goal_function import TextToTextGoalFunction


class TestcaseFailOutputGoalFunction(TextToTextGoalFunction):
    """
    Ensure that the pass@ score has decreased.

    We will use self.ground_truth_output to represent task_id which we correlate with problem.
    """
    DATASET_PARAMS = dict(mini=True, noextreme=True)
    HUMANEVAL = get_human_eval_plus(**DATASET_PARAMS)
    GROUNDTRUTH = get_groundtruth(HUMANEVAL, get_human_eval_plus_hash(**DATASET_PARAMS), [])

    def clear_cache(self):
        if self.use_cache:
            self._call_model_cache.clear()

        get_test_score.cache_clear()

    def _is_goal_complete(self, model_output, attacked_text):
        # We want more than 50% of the testcases to fail!
        return self._get_score(model_output, attacked_text) <= 0.5

    def _get_score(self, model_output, attacked_text):
        eval_score = get_test_score(
            task_id=self.ground_truth_output,
            model_output=model_output
        )
        return eval_score


@functools.lru_cache(maxsize=2 ** 12)
def get_test_score(*, task_id, model_output, is_completion=True):
    problem = TestcaseFailOutputGoalFunction.HUMANEVAL[task_id]
    prompt = problem['prompt']

    if is_completion:
        solution = prompt + model_output
    else:
        solution = model_output

    eval_results = check_correctness(
        dataset='humaneval',
        completion_id=time.time_ns(),
        expected_output=TestcaseFailOutputGoalFunction.GROUNDTRUTH[task_id],
        problem=problem,
        solution=solution,
        base_only=False,
        gt_time_limit_factor=10.0
    )

    total = eval_results['base'][1] + eval_results['plus'][1]
    passed = [i for i in total if i == 1]
    return float(len(passed)) / float(len(total))
