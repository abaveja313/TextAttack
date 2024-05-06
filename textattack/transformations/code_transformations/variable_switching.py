import jedi
import random
import string

from textattack.shared import AttackedText
from textattack.transformations import Transformation


class VariableSwitchingTransformation(Transformation):
    @staticmethod
    def generate_random_string(length):
        return ''.join(random.choices(string.ascii_letters, k=length))

    @staticmethod
    def is_valid_variable(var):
        return (
                var.type == 'statement' and
                not (var.description.startswith('def ') or
                     var.description.startswith('class ') or
                     var.description.startswith('lambda ') or
                     var.description.startswith('import ') or
                     var.description.startswith('param '))
        )

    def _get_transformations(self, attacked_text, indices_to_modify):
        current_text = attacked_text.text
        script = jedi.Script(current_text)
        variables = [var for var in script.get_names(all_scopes=True, definitions=True) if self.is_valid_variable(var)]
        refactorings = []

        for i in range(len(variables)):
            original_names = [var.name for var in variables]
            new_name = self.generate_random_string(8)

            refactored_code = current_text
            for var in variables:
                if var.name == original_names[i]:
                    refactored_code = refactored_code.replace(var.name, new_name)

            refactorings.append(refactored_code)

        return [AttackedText(r) for r in refactorings]

    @property
    def deterministic(self):
        return True
