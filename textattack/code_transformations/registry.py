from enum import Enum, auto


class CRT(Enum):
    lexical = auto()
    syntax = auto()
    semantics = auto()
    optimization = auto()


class CodeTransformationRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, cls, categories):
        for category in categories:
            self._registry.setdefault(category, [])
            self._registry[category].append(cls)

    def get(self, category=None):
        if not category:
            return self._registry.values()

        return self._registry.get(category, [])


class RegistedMixin:
    registry = CodeTransformationRegistry()

    def __init_subclass__(cls, categories=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if not categories or len(categories) == 0:
            categories = ('default',)

        if not isinstance(categories, (list, tuple)):
            categories = (categories,)

        RegistedMixin.registry.register(cls, categories)
