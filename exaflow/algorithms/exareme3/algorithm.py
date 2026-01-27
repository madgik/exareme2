from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import Optional

from exaflow.algorithms.utils.inputdata_utils import Inputdata


class Algorithm(ABC):
    """
    Base class for all exaflow algorithm flows.

    Notes for authors:
    - `algname` must match the `"name"` field of the accompanying <algorithm>.json.
    - `engine` is the algorithm execution engine injected by the framework. It
      exposes methods such as `run_algorithm_udf` to execute the UDFs declared
      in the flow (see `exareme3_registry.exareme3_udf` for registration).
    - `inputdata` is a pydantic model describing the requested variables,
      datasets and filters (see `utils/inputdata_utils.Inputdata`). Unlike
      exareme2, `x`/`y` can be optional in the model; flows are expected to
      validate required fields in their `run` method.
    - `metadata` passed to `run()` is a dictionary keyed by variable name. Today
      flows expect at least `metadata[var]["is_categorical"]` to exist when
      distinguishing categorical vs numerical variables.
    - `parameters` holds the algorithm parameters as provided by the user.
    """

    algname: str

    def __init__(
        self,
        *,
        inputdata: Inputdata,
        engine,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        self._inputdata = inputdata
        self._engine = engine
        self._parameters = parameters if parameters is not None else {}

    def __init_subclass__(cls, algname: str, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.algname = algname

    @property
    def engine(self):
        return self._engine

    @property
    def inputdata(self):
        return self._inputdata

    @property
    def parameters(self) -> Dict[str, Any]:
        return self._parameters

    @abstractmethod
    def run(self, metadata: dict):
        """
        Execute the algorithm flow.

        Parameters
        ----------
        metadata : dict
            Per-variable metadata. Flows rely on at least `metadata[var]["is_categorical"]`
            to decide encoding, and may use additional keys as needed.
        """
        pass
