from collections.abc import Mapping


class AttrDict(dict):
    def __init__(self, dct: Mapping):
        recdict = {
            key: AttrDict(val) if isinstance(val, Mapping) else val
            for key, val in dct.items()
        }
        super(AttrDict, self).__init__(recdict)
        self.__dict__ = self
