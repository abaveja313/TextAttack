import functools
import time

from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
from evalplus.evaluate import get_groundtruth, check_correctness

from textattack.goal_functions.text.text_to_text_goal_function import TextToTextGoalFunction


class TestcaseFailOutputGoalFunction(TextToTextGoalFunction):
    """
    Ensure that the pass@ score has decreased.

    We will use self.ground_truth_output to represent task_id which we correlate with problem.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset_params = dict(mini=True)
        self.human_eval = get_human_eval_plus(**dataset_params)
        self.ground_truth = get_groundtruth(self.human_eval, get_human_eval_plus_hash(**dataset_params), [])

    @functools.lru_cache(maxsize=2 ** 12)
    def get_test_score(self, *, task_id, model_output, stem, is_completion=True):
        problem = self.human_eval[task_id]

        if is_completion:
            solution = stem + model_output
        else:
            solution = model_output

        eval_results = check_correctness(
            dataset='humaneval',
            completion_id=time.time_ns(),
            expected_output=self.ground_truth[task_id],
            problem=problem,
            solution=solution,
            base_only=False,
            gt_time_limit_factor=15.0
        )

        total = eval_results['base'][1] + eval_results['plus'][1]
        passed = [i for i in total if i == 1]

        print("Total", total, ". Passed", passed)

        if len(total) == 0:
            return 1.0
        return float(len(passed)) / float(len(total))

    def clear_cache(self):
        if self.use_cache:
            self._call_model_cache.clear()

        self.get_test_score.cache_clear()

    def _is_goal_complete(self, model_output, attacked_text):
        # We want more than 50% of the testcases to fail!
        return self._get_score(model_output, attacked_text) <= 0.25

    def _get_score(self, model_output, attacked_text):
        eval_score = self.get_test_score(
            task_id=self.ground_truth_output,
            model_output=model_output,
            stem=attacked_text.text
        )
        return eval_score
