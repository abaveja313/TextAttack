import functools
import math
import time

from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
from evalplus.evaluate import get_groundtruth, check_correctness
from textattack.shared.utils import remove_pass
from textattack.goal_functions.text.text_to_text_goal_function import TextToTextGoalFunction
from loguru import logger


class TestcaseFailOutputGoalFunction(TextToTextGoalFunction):
    """
    Ensure that the pass@ score has decreased.

    We will use self.ground_truth_output to represent task_id which we correlate with problem.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset_params = dict(mini=True)
        self.human_eval = get_human_eval_plus(**dataset_params)
        self.ground_truth = get_groundtruth(
            self.human_eval,
            get_human_eval_plus_hash(**dataset_params),
            []
        )

    @functools.lru_cache(maxsize=2 ** 12)
    def get_test_score(self, *, task_id, model_output, stem, is_completion=True):
        problem = self.human_eval[task_id]
        formatted_stem = remove_pass(inputs=[stem])[0]
        if is_completion:
            if not stem.startswith('\n'):
                solution = f"{formatted_stem}\n{model_output}"
            else:
                solution = formatted_stem + model_output
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

        if len(total) == 0:
            logger.warning("Pass Rate=0%, Syntactic Error")
            return 0.0  # Minimum score!

        result = float(len(passed)) / float(len(total))
        logger.info(f"Total={len(total)}, Passed={len(passed)}, Pass Rate={result}")
        return result

    def clear_cache(self):
        if self.use_cache:
            self._call_model_cache.clear()

        self.get_test_score.cache_clear()

    def _is_goal_complete(self, model_output, attacked_text):
        result = math.isclose(self._get_score(model_output, attacked_text), 0.0)
        if result:
            logger.warning("Found output that destroys model!")
            # logger.info(remove_pass(attacked_text.text) + "\n" + model_output)
        return result

    def _get_score(self, model_output, attacked_text):
        eval_score = self.get_test_score(
            task_id=self.ground_truth_output,
            model_output=model_output,
            stem=attacked_text.text
        )
        return -1 * eval_score  # we want to maximize this score in optimization
