from collections import OrderedDict
from typing import Callable, Iterable

from textattack.shared import AttackedText
from textattack.tokenization.lang_processor import LangProcessor
from textattack.tokenization.python_processor import PythonProcessor


class AttackedCode(AttackedText):
    def __init__(self, code_input, attack_attrs=None):
        self.lang_processor: LangProcessor = PythonProcessor()
        super().__init__(code_input, attack_attrs)

    @property
    def tokenizer(self) -> Callable[[str], list[str]]:
        return lambda code: self.lang_processor.tokenize_code(
            code,
            keep_comments=False,
            process_strings=True
        )

    @property
    def words(self) -> list[str]:
        if not self._words:
            self._words = self.tokenizer(self.text)
        return self._words

    def generate_new_attacked_code(self, new_code: str) -> 'AttackedCode':
        new_attack_attrs = dict()
        new_attack_attrs["newly_modified_indices"] = set()
        # Keep track of chain
        new_attack_attrs["previous_attacked_text"] = self
        new_attack_attrs["modified_indices"] = self.attack_attrs["modified_indices"].copy()
        new_attack_attrs["original_index_map"] = self.attack_attrs["original_index_map"].copy()

        new_i = 0
        new_tokens = self.tokenizer(new_code)
        for i, (input_token, adv_token) in enumerate(zip(self.words, new_tokens)):

            if input_token != adv_token:
                new_attack_attrs["modified_indices"].add(new_i)
                new_attack_attrs["newly_modified_indices"].add(new_i)

            new_i += 1

        return AttackedCode(new_code, attack_attrs=new_attack_attrs)

    def __repr__(self) -> str:
        return f'<AttackedCode "{self.text}">'

