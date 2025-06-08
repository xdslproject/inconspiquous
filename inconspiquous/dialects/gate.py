from __future__ import annotations
from typing import ClassVar, Literal

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
from inconspiquous.gates import (
    GateAttr,
    SingleQubitCliffordGate,
    TwoQubitCliffordGate,
)
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
class HadamardGate(SingleQubitCliffordGate):
    name = "gate.h"

    def pauli_prop(
        self, input_idx: int, pauli_type: Literal["X", "Z"]
    ) -> tuple[tuple[bool, bool], ...]:
        assert input_idx == 0
        if pauli_type == "X":
            return ((False, True),)  # X propagates to Z
        else:
            return ((True, False),)  # Z propagates to X


@irdl_attr_definition
class XGate(SingleQubitCliffordGate):
    name = "gate.x"

    def pauli_prop(
        self, input_idx: int, pauli_type: Literal["X", "Z"]
    ) -> tuple[tuple[bool, bool], ...]:
        assert input_idx == 0
        if pauli_type == "X":
            return ((True, False),)  # X propagates to X
        else:
            return ((False, True),)  # Z propagates to Z


@irdl_attr_definition
class YGate(SingleQubitCliffordGate):
    name = "gate.y"

    def pauli_prop(
        self, input_idx: int, pauli_type: Literal["X", "Z"]
    ) -> tuple[tuple[bool, bool], ...]:
        assert input_idx == 0
        if pauli_type == "X":
            return ((False, True),)  # X propagates to Z
        else:
            return ((True, False),)  # Z propagates to X


@irdl_attr_definition
class ZGate(SingleQubitCliffordGate):
    name = "gate.z"

    def pauli_prop(
        self, input_idx: int, pauli_type: Literal["X", "Z"]
    ) -> tuple[tuple[bool, bool], ...]:
        assert input_idx == 0
        if pauli_type == "X":
            return ((True, False),)  # X propagates to X
        else:
            return ((False, True),)  # Z propagates to Z


@irdl_attr_definition
class PhaseGate(SingleQubitCliffordGate):
    name = "gate.s"

    def pauli_prop(
        self, input_idx: int, pauli_type: Literal["X", "Z"]
    ) -> tuple[tuple[bool, bool], ...]:
        assert input_idx == 0
        if pauli_type == "X":
            return ((True, True),)  # X propagates to XZ
        else:
            return ((False, True),)  # Z propagates to Z


@irdl_attr_definition
class PhaseDaggerGate(SingleQubitCliffordGate):
    name = "gate.s_dagger"

    def pauli_prop(
        self, input_idx: int, pauli_type: Literal["X", "Z"]
    ) -> tuple[tuple[bool, bool], ...]:
        assert input_idx == 0
        if pauli_type == "X":
            return ((True, True),)  # X propagates to XZ
        else:
            return ((False, True),)  # Z propagates to Z


@irdl_attr_definition
class TGate(SingleQubitCliffordGate):
    name = "gate.t"

    def pauli_prop(
        self, input_idx: int, pauli_type: Literal["X", "Z"]
    ) -> tuple[tuple[bool, bool], ...]:
        assert input_idx == 0
        if pauli_type == "X":
            return ((True, True),)  # X propagates to XZ
        else:
            return ((False, True),)  # Z propagates to Z


@irdl_attr_definition
class TDaggerGate(SingleQubitCliffordGate):
    name = "gate.t_dagger"

    def pauli_prop(
        self, input_idx: int, pauli_type: Literal["X", "Z"]
    ) -> tuple[tuple[bool, bool], ...]:
        assert input_idx == 0
        if pauli_type == "X":
            return ((True, True),)  # X propagates to XZ
        else:
            return ((False, True),)  # Z propagates to Z


@irdl_attr_definition
class RZGate(SingleQubitCliffordGate):
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

    def pauli_prop(
        self, input_idx: int, pauli_type: Literal["X", "Z"]
    ) -> tuple[tuple[bool, bool], ...]:
        assert input_idx == 0
        if pauli_type == "X":
            return ((True, True),)  # X propagates to XZ
        else:
            return ((False, True),)  # Z propagates to Z


@irdl_attr_definition
class JGate(SingleQubitCliffordGate):
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

    def pauli_prop(
        self, input_idx: int, pauli_type: Literal["X", "Z"]
    ) -> tuple[tuple[bool, bool], ...]:
        assert input_idx == 0
        if pauli_type == "X":
            return ((True, True),)  # X propagates to XZ
        else:
            return ((False, True),)  # Z propagates to Z


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
class CXGate(TwoQubitCliffordGate):
    name = "gate.cx"

    def pauli_prop(
        self, input_idx: int, pauli_type: Literal["X", "Z"]
    ) -> tuple[tuple[bool, bool], ...]:
        if pauli_type == "X":
            if input_idx == 0:
                return (
                    (True, False),
                    (True, False),
                )  # X on first input propagates to X on both outputs
            else:
                return (
                    (False, False),
                    (True, False),
                )  # X on second input propagates to X on second output
        else:
            if input_idx == 0:
                return (
                    (False, True),
                    (False, False),
                )  # Z on first input propagates to Z on first output
            else:
                return (
                    (False, True),
                    (False, True),
                )  # Z on second input propagates to Z on both outputs


@irdl_attr_definition
class CZGate(TwoQubitCliffordGate):
    name = "gate.cz"

    def pauli_prop(
        self, input_idx: int, pauli_type: Literal["X", "Z"]
    ) -> tuple[tuple[bool, bool], ...]:
        if pauli_type == "X":
            if input_idx == 0:
                return (
                    (True, False),
                    (False, True),
                )  # X on first input propagates to X on first output and Z on second
            else:
                return (
                    (False, True),
                    (True, False),
                )  # X on second input propagates to Z on first output and X on second
        else:
            if input_idx == 0:
                return (
                    (False, True),
                    (False, False),
                )  # Z on first input propagates to Z on first output
            else:
                return (
                    (False, False),
                    (False, True),
                )  # Z on second input propagates to Z on second output


@irdl_attr_definition
class ToffoliGate(GateAttr):
    name = "gate.toffoli"

    @property
    def num_qubits(self) -> int:
        return 3


@irdl_attr_definition
class IdentityGate(SingleQubitCliffordGate):
    name = "gate.id"

    def pauli_prop(
        self, input_idx: int, pauli_type: Literal["X", "Z"]
    ) -> tuple[tuple[bool, bool], ...]:
        assert input_idx == 0
        if pauli_type == "X":
            return ((True, False),)  # X propagates to X
        else:
            return ((False, True),)  # Z propagates to Z


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
