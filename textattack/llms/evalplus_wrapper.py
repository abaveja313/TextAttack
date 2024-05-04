from models import VllmDecoder
from textattack.models.wrappers import ModelWrapper


class EvalPlusWrapper(ModelWrapper):
    def __init__(self, model: VllmDecoder):
        self.model = model

    def __call__(self, text_input_list, **kwargs):
        outputs = self.model.multi_codegen(
            prompts=text_input_list,
            do_sample=True
        )

        if len(text_input_list) == 1:
            return outputs[0]
        return outputs
