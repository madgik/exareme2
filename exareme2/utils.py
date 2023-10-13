from collections.abc import Mapping


class AttrDict(dict):
    def __init__(self, dct: Mapping):
        recdict = {
            key: AttrDict(val) if isinstance(val, Mapping) else val
            for key, val in dct.items()
        }
        super(AttrDict, self).__init__(recdict)
        self.__dict__ = self


class Singleton(type):
    """
    Copied from https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
