from xdsl.ir import Dialect
from xdsl.irdl import irdl_attr_definition
from inconspiquous.gates import GateAttr


@irdl_attr_definition
class PerfectCode5QubitCorrectionAttr(GateAttr):
    name = "qec.perfect"

    @property
    def num_qubits(self) -> int:
        return 5


QEC = Dialect(
    "qec",
    [],
    [PerfectCode5QubitCorrectionAttr],
)
