from textattack.code_transformations.mutation import PostprocessingTransformer
from textattack.code_transformations.registry import CRT


class BlockCommentsTransformer(PostprocessingTransformer, categories=CRT.lexical):
    @property
    def comment(self):
        return "# I am a comment\n# I am a comment"

    @property
    def deterministic(self):
        return True

    def transform(self, code: str):
        results = []
        new_lines = code.split('\n')
        for idx in range(len(new_lines)):
            copied = new_lines.copy()
            if new_lines[idx].startswith('\t'):
                indentation_level = len(new_lines[idx]) - len(new_lines[idx].lstrip('\t'))
                indented_comment = "\n".join([f"\t{self.comment}"] * indentation_level)
                copied.insert(idx, indented_comment)
            results.append('\n'.join(copied))
        return results

    @property
    def attack_func(self) -> callable:
        return self.transform
