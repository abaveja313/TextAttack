import pickle
import time

from textattack.truth_initializers.initializer import Initializer
from textattack.truth_initializers.utils import find_centroid_program
from textattack.shared.utils import (
    remove_comments_and_docstrings,
    normalize_indentation,
)
from functools import cached_property
import joblib
import os

import tqdm
from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
from evalplus.evaluate import check_correctness, get_groundtruth

from textattack.llms.evalplus_wrapper import EvalPlusWrapper
from loguru import logger
import numpy as np


class NoPassingSolutionException(Exception):
    pass


class TestPassingInitializer(Initializer):
    def __init__(
        self,
        model: EvalPlusWrapper,
        task_id: str,
        passing_threshold: float = 1.0,
        num_samples: int = 300,
        batch_size: int = 50,
        min_correct_samples: int = 10,
        mini=True,
        noextreme=False,
    ):

        if mini and noextreme:
            raise ValueError("Cannot specify both mini=True and noextreme=True")

        self.dataset_params = dict(mini=mini, noextreme=noextreme)
        self.problem = self.human_eval[task_id]
        self.batch_size = batch_size
        self.passing_threshold = passing_threshold
        self.num_samples = num_samples
        self.model_wrapper = model
        self.min_correct_samples = min_correct_samples
        self.cache_dir = "cache"

    @cached_property
    def human_eval(self) -> dict[str, dict]:
        return get_human_eval_plus(**self.dataset_params)

    @cached_property
    def ground_truth(self):
        return get_groundtruth(
            self.human_eval, get_human_eval_plus_hash(**self.dataset_params), []
        )

    def batch_generate_sequences(self):
        sequences = []
        remaining = self.num_samples

        with tqdm.tqdm(total=self.num_samples, desc="Generating sequences") as pbar:
            while remaining > 0:
                to_gen = min(self.batch_size, remaining)
                samples = self.model_wrapper.model.codegen(
                    prompt=self.problem["prompt"], do_sample=True, num_samples=to_gen
                )
                sequences.extend(samples)
                pbar.update(to_gen)
                remaining -= to_gen

        return sequences

    def postprocess_sequences(self, sequences: list[str]):
        transforms = (
            lambda c: remove_comments_and_docstrings(c, remove_docstrings=True),
            normalize_indentation,
        )
        processed = []
        for sequence in tqdm.tqdm(sequences, desc="Postprocessing Samples"):
            try:
                result = sequence
                for transform in transforms:
                    result = transform(result)
                processed.append(result)
            except Exception:
                logger.exception("Unable to postprocess sequence")
                logger.warning(f"Solution:\n{sequence}")
                continue
        return processed

    def find_centroid_solution(self):
        cache_file = os.path.join(
            self.cache_dir, f'{self.task_id.replace("/", "_")}.pkl'
        )

        if os.path.exists(cache_file):
            logger.info(f"Loading cached centroid solution for task {self.task_id}")
            return joblib.load(cache_file)

        sequences = self.batch_generate_sequences()
        sequences = self.postprocess_sequences(sequences)
        solutions = []
        failed_stats = []

        for sequence in tqdm.tqdm(sequences, desc="Evaluating Sequences"):
            full_solution = self.problem["prompt"] + sequence
            eval_results = check_correctness(
                dataset="humaneval",
                completion_id=time.time_ns(),
                expected_output=self.ground_truth[self.task_id],
                problem=self.problem,
                solution=full_solution,
                base_only=False,
                gt_time_limit_factor=15.0,
            )

            total = eval_results["base"][1] + eval_results["plus"][1]
            if len(total) == 0:
                logger.warning(
                    "No results were found for a syntactically incorrect solution."
                )
                continue

            passed = [i for i in total if i == 1]
            pass_ratio = float(len(passed)) / float(len(total))
            if pass_ratio >= self.passing_threshold:
                solutions.append(full_solution)
            else:
                failed_stats.append(pass_ratio)

        if len(solutions) < self.min_correct_samples:
            raise NoPassingSolutionException(
                f"Needed {self.min_correct_samples} correct solutions, but found {len(solutions)}"
            )

        self._print_failure_stats(failed_stats)
        logger.info(
            f"Found {len(solutions)} correct solutions. Finding AST + Levenshtein Centroid."
        )
        centroid_solution = find_centroid_program(solutions)

        os.makedirs(self.cache_dir, exist_ok=True)

        with open(cache_file, "wb") as sol:
            pickle.dump(centroid_solution, sol)

        return centroid_solution

    def _print_failure_stats(self, fail_rates: list[float]):
        debug_message = [
            "Failure Rate Stats",
            f"Failure Rate: {round(float(len(fail_rates)) / self.num_samples, 4) * 100}%",
            f"Mean: {np.mean(fail_rates)}",
            f"Median: {np.median(fail_rates)}",
            f"Stddev: {np.std(fail_rates)}",
        ]
        logger.info("\n".join(debug_message))
