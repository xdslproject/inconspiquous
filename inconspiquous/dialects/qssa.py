from __future__ import annotations

from xdsl.ir import (
    Dialect,
    Operation,
    ParametrizedAttribute,
    SSAValue,
    TypeAttribute,
    VerifyException,
)
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
    var_operand_def,
    var_result_def,
)
from xdsl.dialects.builtin import IntegerType
from xdsl.parser import Parser
from xdsl.printer import Printer

from inconspiquous.gates import GateAttr


@irdl_attr_definition
class QubitAttr(ParametrizedAttribute, TypeAttribute):
    """
    Type for a single qubit.
    """

    name = "qssa.qubit"


@irdl_op_definition
class AllocOp(IRDLOperation):
    name = "qssa.alloc"

    out = result_def(QubitAttr())

    assembly_format = "attr-dict"

    def __init__(self):
        super().__init__(
            operands=(),
            result_types=[[QubitAttr()]],
        )


@irdl_op_definition
class GateOp(IRDLOperation):
    name = "qssa.gate"

    gate = prop_def(GateAttr)

    ins = var_operand_def(QubitAttr())

    outs = var_result_def(QubitAttr())

    def __init__(self, gate: GateAttr, *ins: Operation | SSAValue):
        super().__init__(
            operands=(ins,),
            result_types=([QubitAttr()] * len(ins),),
            properties={"gate": gate},
        )

    # Assembly format cannot cope with var_results
    @classmethod
    def parse(cls, parser: Parser) -> GateOp:
        with parser.in_angle_brackets():
            gate_attr = parser.parse_attribute()
        args = parser.parse_comma_separated_list(
            Parser.Delimiter.NONE, parser.parse_operand
        )
        attr_dict = parser.parse_optional_attr_dict()
        return GateOp.create(
            properties={"gate": gate_attr},
            operands=args,
            result_types=tuple(QubitAttr() for _ in args),
            attributes=attr_dict,
        )

    def print(self, printer: Printer):
        with printer.in_angle_brackets():
            printer.print_attribute(self.gate)

        printer.print_string(" ")
        printer.print_list(self.ins, printer.print_operand)
        if self.attributes:
            printer.print_string(" ")
            printer.print_attr_dict(self.attributes)

    def verify_(self) -> None:
        num_qubits = self.gate.num_qubits()
        if len(self.ins) != num_qubits:
            raise VerifyException(
                f"expected {num_qubits} qubit inputs, but got {len(self.ins)} instead."
            )
        if len(self.outs) != num_qubits:
            raise VerifyException(
                f"expected {num_qubits} qubit outputs, but got {len(self.outs)} instead."
            )


@irdl_op_definition
class MeasureOp(IRDLOperation):
    name = "qssa.measure"

    qubit = operand_def(QubitAttr())

    out = result_def(IntegerType(1))

    assembly_format = "$qubit attr-dict"

    def __init__(self, qubit: Operation | SSAValue):
        super().__init__(
            operands=(qubit,),
            result_types=(IntegerType(1),),
        )


Qssa = Dialect(
    "qssa",
    [
        AllocOp,
        GateOp,
        MeasureOp,
    ],
    [
        QubitAttr,
    ],
)
