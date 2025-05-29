from __future__ import annotations
from typing import ClassVar
from xdsl.ir import Attribute, Dialect, OpResult, SSAValue
from xdsl.irdl importirdl_attr_definition,irdl_op_definition, ParameterDef, VarOperandDef, builder
from typing import List, Tuple, Literal

from xdsl.dialects.builtin import (
    IndexType,
    IntegerAttr,
    AnyFloatConstr,
    i1,
    IntegerType,
)
from xdsl.ir import (
    Dialect,
    Operation,
    ParametrizedAttribute,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AnyInt,
    BaseAttr,
    GenericAttrConstraint,
    IRDLOperation,
    IntConstraint,
    IntVarConstraint,
    ParamAttrConstraint,
    ParameterDef,
    VarConstraint,
    base,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
    traits_def,
    eq,
)
from xdsl.parser import AttrParser
from xdsl.pattern_rewriter import RewritePattern
from xdsl.printer import Printer
from xdsl.traits import ConstantLike, HasCanonicalizationPatternsTrait, Pure
from xdsl.dialects.builtin import IntAttrConstraint

from inconspiquous.dialects.angle import AngleAttr, AngleType
from inconspiquous.gates import GateAttr, SingleQubitGate, TwoQubitGate
from inconspiquous.constraints import SizedAttributeConstraint


@irdl_attr_definition
class GateType(ParametrizedAttribute, TypeAttribute):
    """
    Type for dynamic gate operations
    """

    name = "gate.type"

    num_qubits: ParameterDef[IntegerAttr[IndexType]]

    def __init__(self, num_qubits: int | IntegerAttr[IndexType]):
        if isinstance(num_qubits, int):
            num_qubits = IntegerAttr.from_index_int_value(num_qubits)
        super().__init__((num_qubits,))

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> tuple[IntegerAttr[IndexType]]:
        with parser.in_angle_brackets():
            i = parser.parse_integer(allow_boolean=False, allow_negative=False)
            return (IntegerAttr.from_index_int_value(i),)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string(str(self.num_qubits.value.data))

    @classmethod
    def constr(
        cls, int_constraint: IntConstraint | None = None
    ) -> GenericAttrConstraint[GateType]:
        if int_constraint is None:
            return BaseAttr(GateType)
        return ParamAttrConstraint(
            GateType,
            (
                IntegerAttr.constr(
                    value=IntAttrConstraint(int_constraint), type=eq(IndexType())
                ),
            ),
        )

@irdl_attr_definition
class CliffordGateAttr(GateAttr):
    """
    Attribute for Clifford gates that defines how Pauli operators propagate through them.
    """
    name = "gate.clifford_gate"

    def pauli_prop(
        self, input_idx: int, pauli_type: Literal["X", "Z"]
    ) -> Tuple[Tuple[bool, bool], ...]:
        """
        Define how a Pauli gate propagates through this Clifford gate.

        Args:
            input_idx: The index of the input qubit that has the Pauli gate.
            pauli_type: Either "X" or "Z" to indicate which Pauli gate.

        Returns:
            A tuple of pairs (has_x, has_z) for each output qubit, where:
              - has_x is True if an X gate should be applied to that output.
              - has_z is True if a Z gate should be applied to that output.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} is a Clifford gate "
            "but does not implement pauli_prop"
        )



@irdl_op_definition
class ConstantGateOp(IRDLOperation):
    """
    Constant-like operation for producing gates
    """

    _I: ClassVar = IntVarConstraint("I", AnyInt())

    name = "gate.constant"

    gate = prop_def(SizedAttributeConstraint(GateAttr, _I))

    out = result_def(GateType.constr(_I))

    assembly_format = "$gate attr-dict"

    traits = traits_def(
        ConstantLike(),
        Pure(),
    )

    def __init__(self, gate: GateAttr):
        super().__init__(
            properties={
                "gate": gate,
            },
            result_types=(GateType(gate.num_qubits),),
        )


@irdl_attr_definition
class HadamardGate(CliffordGateAttr):
    name = "gate.h"
    def pauli_prop(
        self, input_idx: int, pauli_type: Literal["X", "Z"]
    ) -> Tuple[Tuple[bool, bool], ...]:
        if input_idx != 0:
            raise ValueError("HadamardGate only has one input qubit.")
        if pauli_type == "X":
            return ((False, True),)  # X -> Z
        if pauli_type == "Z":
            return ((True, False),)  # Z -> X
        raise ValueError(f"Unknown Pauli type: {pauli_type}")



@irdl_attr_definition
class XGate(SingleQubitGate):
    name = "gate.x"


@irdl_attr_definition
class YGate(SingleQubitGate):
    name = "gate.y"


@irdl_attr_definition
class ZGate(SingleQubitGate):
    name = "gate.z"


@irdl_attr_definition
class PhaseGate(SingleQubitGate):
    name = "gate.s"


@irdl_attr_definition
class PhaseDaggerGate(SingleQubitGate):
    name = "gate.s_dagger"


@irdl_attr_definition
class TGate(SingleQubitGate):
    name = "gate.t"


@irdl_attr_definition
class TDaggerGate(SingleQubitGate):
    name = "gate.t_dagger"


@irdl_attr_definition
class RZGate(SingleQubitGate):
    name = "gate.rz"

    angle: ParameterDef[AngleAttr]

    def __init__(self, angle: float | AngleAttr):
        if not isinstance(angle, AngleAttr):
            angle = AngleAttr(angle)

        super().__init__((angle,))

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> tuple[AngleAttr]:
        return (AngleAttr.new(AngleAttr.parse_parameters(parser)),)

    def print_parameters(self, printer: Printer) -> None:
        return self.angle.print_parameters(printer)


@irdl_attr_definition
class JGate(SingleQubitGate):
    name = "gate.j"

    angle: ParameterDef[AngleAttr]

    def __init__(self, angle: float | AngleAttr):
        if not isinstance(angle, AngleAttr):
            angle = AngleAttr(angle)

        super().__init__((angle,))

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> tuple[AngleAttr]:
        return (AngleAttr.new(AngleAttr.parse_parameters(parser)),)

    def print_parameters(self, printer: Printer) -> None:
        return self.angle.print_parameters(printer)


class DynJGateHasCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from inconspiquous.transforms.canonicalization import gate

        return (gate.DynJGateToJPattern(),)


@irdl_op_definition
class DynJGate(IRDLOperation):
    name = "gate.dyn_j"

    angle = operand_def(AngleType)

    out = result_def(GateType(1))

    traits = traits_def(Pure(), DynJGateHasCanonicalizationPatterns())

    assembly_format = "`` `<` $angle `>` attr-dict"

    def __init__(self, angle: SSAValue | Operation):
        super().__init__(operands=(angle,), result_types=(GateType(1),))


@irdl_attr_definition
class CXGate(CliffordGateAttr):
    name = "gate.cx"
    def pauli_prop(
        self, input_idx: int, pauli_type: Literal["X", "Z"]
    ) -> Tuple[Tuple[bool, bool], ...]:
        if input_idx not in (0, 1):
            raise ValueError("CXGate input_idx must be 0 or 1.")
        if pauli_type == "X":
            if input_idx == 0:  # X on control
                return ((True, False), (True, False))  # X on ctl_out, X on tgt_out
            # input_idx == 1, X on target
            return ((False, False), (True, False))  # I on ctl_out, X on tgt_out
        if pauli_type == "Z":
            if input_idx == 0:  # Z on control
                return ((False, True), (False, False))  # Z on ctl_out, I on tgt_out
            # input_idx == 1, Z on target
            return ((False, True), (False, True))  # Z on ctl_out, Z on tgt_out
        raise ValueError(f"Unknown Pauli type: {pauli_type}")


@irdl_attr_definition
class CZGate(TwoQubitGate):
    name = "gate.cz"
    def pauli_prop(
        self, input_idx: int, pauli_type: Literal["X", "Z"]
    ) -> Tuple[Tuple[bool, bool], ...]:
        if input_idx not in (0, 1):
            raise ValueError("CZGate input_idx must be 0 or 1.")
        if pauli_type == "X":
            if input_idx == 0:  # X on first qubit
                return ((True, False), (False, True))  # X on q0_out, Z on q1_out
            # input_idx == 1, X on second qubit
            return ((False, True), (True, False))  # Z on q0_out, X on q1_out
        if pauli_type == "Z":
            if input_idx == 0:  # Z on first qubit
                return ((False, True), (False, False))  # Z on q0_out, I on q1_out
            # input_idx == 1, Z on second qubit
            return ((False, False), (False, True))  # I on q0_out, Z on q1_out
        raise ValueError(f"Unknown Pauli type: {pauli_type}")



@irdl_attr_definition
class ToffoliGate(GateAttr):
    name = "gate.toffoli"

    @property
    def num_qubits(self) -> int:
        return 3


@irdl_attr_definition
class IdentityGate(SingleQubitGate):
    name = "gate.id"


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


class XZSOpHasCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from inconspiquous.transforms.canonicalization.gate import XZSToXZPattern

        return (XZSToXZPattern(),)


@irdl_op_definition
class XZSOp(IRDLOperation):
    """
    A gadget for describing combinations of X, Z, and (pi/2) phase gates.
    """

    name = "gate.xzs"

    x = operand_def(i1)
    z = operand_def(i1)
    phase = operand_def(i1)

    out = result_def(GateType(1))

    assembly_format = "$x `,` $z `,` $phase attr-dict"

    traits = traits_def(Pure(), XZSOpHasCanonicalizationPatterns())

    def __init__(
        self,
        x: Operation | SSAValue,
        z: Operation | SSAValue,
        phase: Operation | SSAValue,
    ):
        super().__init__(operands=(x, z, phase), result_types=(GateType(1),))


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
    [ConstantGateOp, QuaternionGateOp, ComposeGateOp, XZSOp, XZOp, DynJGate],
    [
        HadamardGate,
        XGate,
        YGate,
        ZGate,
        PhaseGate,
        PhaseDaggerGate,
        TGate,
        TDaggerGate,
        RZGate,
        JGate,
        CXGate,
        CZGate,
        ToffoliGate,
        IdentityGate,
        GateType,
    ],
)
