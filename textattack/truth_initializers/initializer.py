from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np
from loguru import logger
from tqdm import tqdm

from textattack.llms.evalplus_wrapper import EvalPlusWrapper
from textattack.shared.utils import remove_comments_and_docstrings, normalize_indentation


class Initializer(ABC):
    def __init__(
            self,
            model: EvalPlusWrapper,
            task_id: str,
            prompt: str,
            passing_threshold: float,
            num_samples: int = 300,
            min_correct_samples: int = 10,
            batch_size: int = 50
    ):
        self.model = model
        self.task_id = task_id
        self.num_samples = num_samples
        self.prompt = prompt
        self.batch_size = batch_size
        self.passing_threshold = passing_threshold
        self.min_correct_samples = min_correct_samples

    @abstractmethod
    def _canonical_solution(self):
        pass

    @cached_property
    def canonical_solution(self):
        return self._canonical_solution()

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

    def _print_failure_stats(self, fail_rates: list[float]):
        debug_message = [
            "Failure Rate Stats",
            f"Failure Rate: {round(float(len(fail_rates)) / self.num_samples, 4) * 100}%",
            f"Mean: {np.mean(fail_rates)}",
            f"Median: {np.median(fail_rates)}",
            f"Stddev: {np.std(fail_rates)}",
        ]
        logger.info("\n".join(debug_message))
