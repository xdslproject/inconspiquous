from __future__ import annotations
from typing import ClassVar
from abc import ABC, abstractmethod

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


class CliffordGateAttr(GateAttr, ABC):
    """
    Abstract base class for Clifford gates that defines how Pauli operators
    propagate through the gate.
    """

    @abstractmethod
    def pauli_prop(
        self, input_index: int, pauli_type: str
    ) -> tuple[tuple[bool, bool], ...]:
        """
        Determines how a Pauli operator propagates through this gate.

        Args:
            input_index: The index of the input qubit where the Pauli gate is applied.
            pauli_type: Either "X" or "Z", the type of Pauli gate.

        Returns:
            A tuple of tuples. Each inner tuple (bool, bool) represents the
            (X_component, Z_component) of the Pauli operator on the corresponding
            output qubit.

            Example: ((True, False), (False, True)) means X on the first output
            qubit and Z on the second output qubit.
        """
        pass


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
class HadamardGate(SingleQubitGate, CliffordGateAttr):
    name = "gate.h"

    def pauli_prop(
        self, input_index: int, pauli_type: str
    ) -> tuple[tuple[bool, bool], ...]:
        if input_index != 0:
            raise ValueError(
                "HadamardGate is a single-qubit gate, input_index must be 0."
            )

        if pauli_type == "X":
            # X gate transforms to Z: X→H = H→Z
            return ((False, True),)
        elif pauli_type == "Z":
            # Z gate transforms to X: Z→H = H→X
            return ((True, False),)
        else:
            raise ValueError(f"pauli_type must be 'X' or 'Z', got {pauli_type}")


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
class CXGate(TwoQubitGate, CliffordGateAttr):
    name = "gate.cx"

    def pauli_prop(
        self, input_index: int, pauli_type: str
    ) -> tuple[tuple[bool, bool], ...]:
        if input_index not in (0, 1):
            raise ValueError("CXGate is a two-qubit gate, input_index must be 0 or 1.")

        if pauli_type == "X":
            if input_index == 0:
                # X on control propagates to X on both outputs
                return ((True, False), (True, False))
            else:  # input_index == 1
                # X on target propagates to X only on target output
                return ((False, False), (True, False))
        elif pauli_type == "Z":
            if input_index == 0:
                # Z on control propagates to Z only on control output
                return ((False, True), (False, False))
            else:  # input_index == 1
                # Z on target propagates to Z on both outputs
                return ((False, True), (False, True))
        else:
            raise ValueError(f"pauli_type must be 'X' or 'Z', got {pauli_type}")


@irdl_attr_definition
class CZGate(TwoQubitGate, CliffordGateAttr):
    name = "gate.cz"

    def pauli_prop(
        self, input_index: int, pauli_type: str
    ) -> tuple[tuple[bool, bool], ...]:
        if input_index not in (0, 1):
            raise ValueError("CZGate is a two-qubit gate, input_index must be 0 or 1.")

        if pauli_type == "X":
            if input_index == 0:
                # X on first qubit propagates to X on first, Z on second
                return ((True, False), (False, True))
            else:  # input_index == 1
                # X on second qubit propagates to Z on first, X on second
                return ((False, True), (True, False))
        elif pauli_type == "Z":
            if input_index == 0:
                # Z on first qubit propagates to Z on first only
                return ((False, True), (False, False))
            else:  # input_index == 1
                # Z on second qubit propagates to Z on second only
                return ((False, False), (False, True))
        else:
            raise ValueError(f"pauli_type must be 'X' or 'Z', got {pauli_type}")


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
