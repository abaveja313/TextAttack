from textattack.llms.models import GeneralVllmDecoder
from textattack.models.wrappers import ModelWrapper
from typing import List

from loguru import logger


class EvalPlusWrapper(ModelWrapper):
    def __init__(self, model: GeneralVllmDecoder):
        self.model = model

    def preprocess(self, inputs: List[str]) -> List[str]:
        results = []
        for prompt in inputs:
            prompt_lines = prompt.strip().splitlines()
            if prompt_lines[-1].strip() == 'pass':
                logger.info("Removing `pass` from the end of input")
                results.append('\n'.join(prompt_lines[:-1]))
            else:
                results.append(prompt)

        return results

    def __call__(self, text_input_list, **kwargs):
        preprocessed = self.preprocess(text_input_list)
        outputs = self.model.complete_stems(
            prompts=preprocessed,
            do_sample=True
        )

        if len(text_input_list) == 1:
            return outputs[0]
        return outputs
