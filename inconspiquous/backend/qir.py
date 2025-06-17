from dataclasses import dataclass, field
from typing import Dict

from xdsl.ir import SSAValue

import importlib.util

pyqir_available = importlib.util.find_spec("pyqir") is not None

if pyqir_available:
    pass

__all__ = ["QIRBackend", "pyqir_available"]


@dataclass
class QIRBackend:
    """Backend for converting QIR dialect to QIR assembly using PyQIR"""

    qubit_map: Dict[SSAValue, int] = field(default_factory=dict[SSAValue, int])
