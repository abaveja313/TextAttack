from textattack.llms.models import GeneralVllmDecoder
from textattack.models.wrappers import ModelWrapper

from textattack.shared.utils import remove_pass


class EvalPlusWrapper(ModelWrapper):
    def __init__(self, model: GeneralVllmDecoder):
        self.model = model

    def __call__(self, text_input_list, **kwargs):
        preprocessed = remove_pass(inputs=text_input_list)
        outputs = self.model.complete_stems(prompts=preprocessed, do_sample=True)

        if len(text_input_list) == 1:
            return outputs[0]
        return outputs
