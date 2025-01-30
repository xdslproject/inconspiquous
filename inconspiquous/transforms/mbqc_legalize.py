from xdsl.dialects import builtin
from xdsl.dialects.func import FuncOp
from xdsl.ir import Block, Operation
from xdsl.traits import IsTerminator
from xdsl.context import Context
from xdsl.passes import ModulePass
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.utils.exceptions import PassFailedException

from inconspiquous.dialects.angle import CondNegateAngleOp, ConstantAngleOp
from inconspiquous.dialects.measurement import XYDynMeasurementOp, XYMeasurementAttr
from inconspiquous.dialects.qssa import DynGateOp, DynMeasureOp, MeasureOp, GateOp
from inconspiquous.dialects.gate import (
    CZGate,
    ConstantGateOp,
    IdentityGate,
    XGate,
    YGate,
    ZGate,
    XZOp,
)
from inconspiquous.dialects.qubit import AllocOp


class MBQCLegalize(ModulePass):
    """
    Legalize a measurement-based quantum computing program.

    This pass checks that the only operations are:
    - allocation (as zero or plus state)
    - CZ and Pauli gates
    - XZ gadgets
    - measurement in XY plane

    Also moves allocations and CZ gates to the start of the program,
    and any dynamic gates to the end, leaving measurements in the middle.

    Throws an exception if legalization is not possible.
    """

    name = "mbqc-legalize"

    def apply_to_block(self, block: Block):
        # If the block only contains functions, we recurse.
        if all(isinstance(op, FuncOp) for op in block.ops):
            for op in block.ops:
                for region in op.regions:
                    for block in region.blocks:
                        self.apply_to_block(block)
            return

        alloc_ops = list[Operation]()
        cz_ops = list[Operation]()
        correction_ops = list[Operation]()

        current_op = block.first_op

        while current_op is not None:
            next_op = current_op.next_op
            match current_op:
                case AllocOp():
                    alloc_ops.append(current_op)
                case GateOp():
                    match current_op.gate:
                        case CZGate():
                            for operand in current_op.ins:
                                if not (
                                    isinstance(operand.owner, AllocOp)
                                    or (
                                        isinstance(operand.owner, GateOp)
                                        and isinstance(operand.owner.gate, CZGate)
                                    )
                                    or isinstance(operand.owner, Block)
                                ):
                                    raise PassFailedException(
                                        "A CZ gate can only follow allocations and CZ gates in a valid mbqc program."
                                    )
                            cz_ops.append(current_op)
                        case XGate() | YGate() | ZGate():
                            correction_ops.append(current_op)
                        case g:
                            raise PassFailedException(
                                f"Expected only CZ or Pauli gates, found {g}"
                            )
                case DynGateOp():
                    correction_ops.append(current_op)
                case MeasureOp():
                    if not isinstance(current_op.measurement, XYMeasurementAttr):
                        raise PassFailedException(
                            f"Expected only XY measurements, found {current_op.measurement}"
                        )
                    operand = current_op.in_qubits[0]
                    if not (
                        isinstance(operand.owner, AllocOp)
                        or (
                            isinstance(operand.owner, GateOp)
                            and isinstance(operand.owner.gate, CZGate)
                        )
                        or isinstance(operand.owner, Block)
                    ):
                        raise PassFailedException(
                            "A measurement can only follow allocations and CZ gates in a valid mbqc program."
                        )
                case DynMeasureOp():
                    operand = current_op.in_qubits[0]
                    if not (
                        isinstance(operand.owner, AllocOp)
                        or (
                            isinstance(operand.owner, GateOp)
                            and isinstance(operand.owner.gate, CZGate)
                        )
                        or isinstance(operand.owner, Block)
                    ):
                        raise PassFailedException(
                            "A measurement can only follow allocations and CZ gates in a valid mbqc program."
                        )
                case ConstantGateOp():
                    if not (
                        isinstance(
                            current_op.gate, XGate | YGate | ZGate | IdentityGate
                        )
                    ):
                        raise PassFailedException(
                            f"Only expected dynamic Pauli gates, found {current_op.gate}"
                        )
                case (
                    ConstantAngleOp()
                    | XZOp()
                    | CondNegateAngleOp()
                    | XYDynMeasurementOp()
                ):
                    pass
                case o:
                    if o.dialect_name() not in ("arith", "func"):
                        raise PassFailedException(
                            f"Unexpected operation {current_op.name}"
                        )
            current_op = next_op

        for op in (*alloc_ops, *cz_ops, *correction_ops):
            op.detach()
        Rewriter.insert_op((*alloc_ops, *cz_ops), InsertPoint.at_start(block))
        if (term := block.last_op) is not None and term.has_trait(IsTerminator):
            Rewriter.insert_op(correction_ops, InsertPoint.before(term))
        else:
            Rewriter.insert_op(correction_ops, InsertPoint.at_end(block))

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        self.apply_to_block(op.body.block)
