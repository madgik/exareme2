import toml
from importlib.resources import open_text

import mipengine
from mipengine.utils import AttrDict

with open_text(mipengine, "config.toml") as fp:
    config = AttrDict(toml.load(fp))
