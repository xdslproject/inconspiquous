from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Literal

from xdsl.dialects.builtin import (
    AnyFloatConstr,
    IntAttr,
    IntAttrConstraint,
    IntegerType,
    i1,
)
from xdsl.interfaces import HasCanonicalizationPatternsInterface, HasFolderInterface
from xdsl.ir import (
    Dialect,
    Operation,
    ParametrizedAttribute,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AnyInt,
    AttrConstraint,
    BaseAttr,
    IntConstraint,
    IntVarConstraint,
    IRDLOperation,
    ParamAttrConstraint,
    VarConstraint,
    base,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    param_def,
    prop_def,
    result_def,
    traits_def,
)
from xdsl.parser import AttrParser
from xdsl.pattern_rewriter import RewritePattern
from xdsl.printer import Printer
from xdsl.traits import ConstantLike, Pure

from inconspiquous.constraints import SizedAttributeConstraint
from inconspiquous.dialects.angle import AngleAttr, AngleType
from inconspiquous.gates import (
    GateAttr,
    SingleQubitCliffordGate,
    SingleQubitGate,
    TwoQubitCliffordGate,
)
from inconspiquous.gates.core import (
    CliffordGateAttr,
    PauliGate,
    PauliProp,
    TwoQubitGate,
)


@irdl_attr_definition
class GateType(ParametrizedAttribute, TypeAttribute):
    """
    Type for dynamic gate operations
    """

    name = "gate.type"

    num_qubits: IntAttr

    def __init__(self, num_qubits: int | IntAttr):
        if isinstance(num_qubits, int):
            num_qubits = IntAttr(num_qubits)
        super().__init__(num_qubits)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> tuple[IntAttr]:
        with parser.in_angle_brackets():
            i = parser.parse_integer(allow_boolean=False, allow_negative=False)
            return (IntAttr(i),)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_int(self.num_qubits.data)

    @classmethod
    def constr(
        cls, int_constraint: IntConstraint | None = None
    ) -> AttrConstraint[GateType]:
        if int_constraint is None:
            return BaseAttr(GateType)
        return ParamAttrConstraint(
            GateType,
            (IntAttrConstraint(int_constraint),),
        )


@irdl_op_definition
class ConstantGateOp(IRDLOperation, HasFolderInterface):
    """
    Constant-like operation for producing gates
    """

    _I: ClassVar = IntVarConstraint("I", AnyInt())

    name = "gate.constant"

    gate = prop_def(SizedAttributeConstraint(GateAttr, _I))

    out = result_def(GateType.constr(_I))

    assembly_format = "$gate attr-dict"

    traits = traits_def(Pure(), ConstantLike())

    def __init__(self, gate: GateAttr):
        super().__init__(
            properties={
                "gate": gate,
            },
            result_types=(GateType(gate.num_qubits),),
        )

    def fold(self) -> tuple[GateAttr]:
        return (self.gate,)


@irdl_attr_definition
class HadamardGate(SingleQubitCliffordGate):
    name = "gate.h"

    def pauli_prop(
        self, input_idx: int, pauli_type: Literal["X", "Z"]
    ) -> tuple[PauliProp, ...]:
        assert input_idx == 0
        if pauli_type == "X":
            return (PauliProp.Z(),)
        else:
            return (PauliProp.X(),)


@irdl_attr_definition
class XGate(PauliGate):
    name = "gate.x"


@irdl_attr_definition
class YGate(PauliGate):
    name = "gate.y"


@irdl_attr_definition
class ZGate(PauliGate):
    name = "gate.z"


@irdl_attr_definition
class PhaseGate(SingleQubitCliffordGate):
    name = "gate.s"

    def pauli_prop(
        self, input_idx: int, pauli_type: Literal["X", "Z"]
    ) -> tuple[PauliProp, ...]:
        assert input_idx == 0
        if pauli_type == "X":
            return (PauliProp.Y(),)
        else:
            return (PauliProp.Z(),)


@irdl_attr_definition
class PhaseDaggerGate(SingleQubitCliffordGate):
    name = "gate.s_dagger"

    def pauli_prop(
        self, input_idx: int, pauli_type: Literal["X", "Z"]
    ) -> tuple[PauliProp, ...]:
        assert input_idx == 0
        if pauli_type == "X":
            return (PauliProp.Y(),)
        else:
            return (PauliProp.Z(),)


@irdl_attr_definition
class TGate(SingleQubitGate):
    name = "gate.t"


@irdl_attr_definition
class TDaggerGate(SingleQubitGate):
    name = "gate.t_dagger"


class RotationGate(GateAttr, ABC):
    """
    A gate with angle parameter
    """

    angle: AngleAttr

    def __init__(self, angle: float | AngleAttr):
        if not isinstance(angle, AngleAttr):
            angle = AngleAttr(angle)

        super().__init__(angle)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> tuple[AngleAttr]:
        return (AngleAttr.new(AngleAttr.parse_parameters(parser)),)

    def print_parameters(self, printer: Printer) -> None:
        return self.angle.print_parameters(printer)


@irdl_attr_definition
class RXGate(RotationGate, SingleQubitGate):
    name = "gate.rx"

    def __init__(self, angle: float | AngleAttr):
        super().__init__(angle)


@irdl_attr_definition
class RYGate(RotationGate, SingleQubitGate):
    name = "gate.ry"

    def __init__(self, angle: float | AngleAttr):
        super().__init__(angle)


@irdl_attr_definition
class RZGate(RotationGate, SingleQubitGate):
    name = "gate.rz"

    def __init__(self, angle: float | AngleAttr):
        super().__init__(angle)


@irdl_attr_definition
class JGate(RotationGate, SingleQubitGate):
    name = "gate.j"

    def __init__(self, angle: float | AngleAttr):
        super().__init__(angle)


@irdl_attr_definition
class CRXGate(RotationGate, TwoQubitGate):
    name = "gate.crx"

    def __init__(self, angle: float | AngleAttr):
        super().__init__(angle)


@irdl_attr_definition
class CRYGate(RotationGate, TwoQubitGate):
    name = "gate.cry"

    def __init__(self, angle: float | AngleAttr):
        super().__init__(angle)


@irdl_attr_definition
class CRZGate(RotationGate, TwoQubitGate):
    name = "gate.crz"

    def __init__(self, angle: float | AngleAttr):
        super().__init__(angle)


@irdl_attr_definition
class RZZGate(RotationGate, TwoQubitGate):
    name = "gate.rzz"

    def __init__(self, angle: float | AngleAttr):
        super().__init__(angle)


class DynRotationGate(IRDLOperation, HasCanonicalizationPatternsInterface, ABC):
    angle = operand_def(AngleType)

    traits = traits_def(Pure())

    assembly_format = "`` `<` $angle `>` attr-dict"

    @classmethod
    @abstractmethod
    def static_gate(cls) -> type[RotationGate]:
        """Type of the corresponding static gate attribute"""
        ...

    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from inconspiquous.transforms.canonicalization import gate

        return (gate.DynRotationGateToRotationPattern(cls.static_gate()),)


class SingleQubitDynRotationGate(DynRotationGate, ABC):
    out = result_def(GateType(1))

    def __init__(self, angle: SSAValue | Operation):
        super().__init__(operands=(angle,), result_types=(GateType(1),))


class TwoQubitDynRotationGate(DynRotationGate, ABC):
    out = result_def(GateType(2))

    def __init__(self, angle: SSAValue | Operation):
        super().__init__(operands=(angle,), result_types=(GateType(2),))


@irdl_op_definition
class DynRXGate(SingleQubitDynRotationGate):
    name = "gate.dyn_rx"

    @classmethod
    def static_gate(cls) -> type[RotationGate]:
        return RXGate


@irdl_op_definition
class DynRYGate(SingleQubitDynRotationGate):
    name = "gate.dyn_ry"

    @classmethod
    def static_gate(cls) -> type[RotationGate]:
        return RYGate


@irdl_op_definition
class DynRZGate(SingleQubitDynRotationGate):
    name = "gate.dyn_rz"

    @classmethod
    def static_gate(cls) -> type[RotationGate]:
        return RZGate


@irdl_op_definition
class DynJGate(SingleQubitDynRotationGate):
    name = "gate.dyn_j"

    @classmethod
    def static_gate(cls) -> type[RotationGate]:
        return JGate


@irdl_op_definition
class DynCRXGate(TwoQubitDynRotationGate):
    name = "gate.dyn_crx"

    @classmethod
    def static_gate(cls) -> type[RotationGate]:
        return CRXGate


@irdl_op_definition
class DynCRYGate(TwoQubitDynRotationGate):
    name = "gate.dyn_cry"

    @classmethod
    def static_gate(cls) -> type[RotationGate]:
        return CRYGate


@irdl_op_definition
class DynCRZGate(TwoQubitDynRotationGate):
    name = "gate.dyn_crz"

    @classmethod
    def static_gate(cls) -> type[RotationGate]:
        return CRZGate


@irdl_op_definition
class DynRZZGate(TwoQubitDynRotationGate):
    name = "gate.dyn_rzz"

    @classmethod
    def static_gate(cls) -> type[RotationGate]:
        return RZZGate


@irdl_attr_definition
class CXGate(TwoQubitCliffordGate):
    name = "gate.cx"

    def pauli_prop(
        self, input_idx: int, pauli_type: Literal["X", "Z"]
    ) -> tuple[PauliProp, ...]:
        if pauli_type == "X":
            if input_idx == 0:
                return (PauliProp.X(), PauliProp.X())
            else:
                return (
                    PauliProp.none(),
                    PauliProp.X(),
                )
        else:
            if input_idx == 0:
                return (
                    PauliProp.Z(),
                    PauliProp.none(),
                )
            else:
                return (PauliProp.Z(), PauliProp.Z())


@irdl_attr_definition
class CZGate(TwoQubitCliffordGate):
    name = "gate.cz"

    def pauli_prop(
        self, input_idx: int, pauli_type: Literal["X", "Z"]
    ) -> tuple[PauliProp, ...]:
        if pauli_type == "X":
            if input_idx == 0:
                return (
                    PauliProp.X(),
                    PauliProp.Z(),
                )
            else:
                return (
                    PauliProp.Z(),
                    PauliProp.X(),
                )
        else:
            if input_idx == 0:
                return (PauliProp.Z(), PauliProp.none())
            else:
                return (
                    PauliProp.none(),
                    PauliProp.Z(),
                )


@irdl_attr_definition
class ToffoliGate(GateAttr):
    name = "gate.toffoli"

    @property
    def num_qubits(self) -> int:
        return 3


@irdl_attr_definition
class IdentityGate(CliffordGateAttr):
    name = "gate.id"

    qubits: IntAttr = param_def(converter=IntAttr.get)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> tuple[IntAttr]:
        with parser.in_angle_brackets():
            i = parser.parse_integer(allow_boolean=False, allow_negative=False)
            return (IntAttr(i),)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_int(self.qubits.data)

    @property
    def num_qubits(self) -> int:
        return self.qubits.data

    def pauli_prop(self, input_idx: int, pauli_type: Literal["X", "Z"]):
        return tuple(
            PauliProp.from_lit(pauli_type) if i == input_idx else PauliProp.none()
            for i in range(self.qubits.data)
        )


@irdl_op_definition
class QuaternionGateOp(IRDLOperation):
    """
    A gate described by a quaternion.

    The action of the gate on the Bloch sphere is given by the rotation generated
    by conjugating by the quaternion.
    """

    _T: ClassVar = VarConstraint("T", base(IntegerType) | AnyFloatConstr)

    name = "gate.quaternion"

    real = operand_def(_T)
    i = operand_def(_T)
    j = operand_def(_T)
    k = operand_def(_T)

    out = result_def(GateType(1))

    assembly_format = (
        "`<` type($real) `>` $real `+` $i `i` `+` $j `j` `+` $k `k` attr-dict"
    )

    traits = traits_def(Pure())

    def __init__(
        self,
        real: Operation | SSAValue,
        i: Operation | SSAValue,
        j: Operation | SSAValue,
        k: Operation | SSAValue,
    ):
        real = SSAValue.get(real)
        super().__init__(
            operands=(real, i, j, k),
            result_types=(real.type,),
        )


@irdl_op_definition
class ComposeGateOp(IRDLOperation):
    name = "gate.compose"

    _T: ClassVar = VarConstraint("T", base(GateType))

    lhs = operand_def(_T)
    rhs = operand_def(_T)

    out = result_def(_T)

    assembly_format = "$lhs `,` $rhs attr-dict `:` type($out)"

    traits = traits_def(Pure())

    def __init__(self, lhs: SSAValue | Operation, rhs: SSAValue | Operation):
        lhs = SSAValue.get(lhs)
        super().__init__(operands=(lhs, rhs), result_types=(lhs.type,))


@irdl_op_definition
class XZSOp(IRDLOperation, HasCanonicalizationPatternsInterface):
    """
    A gadget for describing combinations of X, Z, and (pi/2) phase gates.
    """

    name = "gate.xzs"

    x = operand_def(i1)
    z = operand_def(i1)
    phase = operand_def(i1)

    out = result_def(GateType(1))

    assembly_format = "$x `,` $z `,` $phase attr-dict"

    traits = traits_def(Pure())

    def __init__(
        self,
        x: Operation | SSAValue,
        z: Operation | SSAValue,
        phase: Operation | SSAValue,
    ):
        super().__init__(operands=(x, z, phase), result_types=(GateType(1),))

    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from inconspiquous.transforms.canonicalization.gate import XZSToXZPattern

        return (XZSToXZPattern(),)


@irdl_op_definition
class XZOp(IRDLOperation):
    """
    A gadget for describing combinations of X and Z gates.
    """

    name = "gate.xz"

    x = operand_def(i1)
    z = operand_def(i1)

    out = result_def(GateType(1))

    assembly_format = "$x `,` $z attr-dict"

    traits = traits_def(Pure())

    def __init__(
        self,
        x: Operation | SSAValue,
        z: Operation | SSAValue,
    ):
        super().__init__(operands=(x, z), result_types=(GateType(1),))


Gate = Dialect(
    "gate",
    [
        ConstantGateOp,
        QuaternionGateOp,
        ComposeGateOp,
        XZSOp,
        XZOp,
        DynRXGate,
        DynRYGate,
        DynRZGate,
        DynJGate,
        DynCRXGate,
        DynCRYGate,
        DynCRZGate,
        DynRZZGate,
    ],
    [
        HadamardGate,
        XGate,
        YGate,
        ZGate,
        PhaseGate,
        PhaseDaggerGate,
        TGate,
        TDaggerGate,
        RXGate,
        RYGate,
        RZGate,
        JGate,
        CRXGate,
        CRYGate,
        CRZGate,
        RZZGate,
        CXGate,
        CZGate,
        ToffoliGate,
        IdentityGate,
        GateType,
    ],
)
