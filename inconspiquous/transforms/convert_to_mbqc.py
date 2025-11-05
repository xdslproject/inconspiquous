from xdsl.dialects import builtin
from xdsl.context import Context
from xdsl.passes import ModulePass
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.common_subexpression_elimination import (
    CommonSubexpressionElimination,
)

from inconspiquous.transforms.convert_to_cme import ToCMEPass
from inconspiquous.transforms.mbqc_legalize import MBQCLegalize
from inconspiquous.transforms.xzs.convert_to_xzs import ConvertToXZS
from inconspiquous.transforms.xzs.commute import XZCommute
from inconspiquous.transforms.xzs.select import XZSSelect


class ToMBQC(ModulePass):
    """
    Converts a circuit to mbqc.

    Assumes that the circuit only contains J and CZ gates.
    """

    name = "convert-to-mbqc"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for p in (
            ToCMEPass(),
            ConvertToXZS(),
            XZSSelect(),
            XZCommute(),
            CommonSubexpressionElimination(),
            CanonicalizePass(),
            CommonSubexpressionElimination(),
            MBQCLegalize(),
        ):
            p.apply(ctx, op)
