from xdsl.dialects import builtin
from xdsl.context import Context
from xdsl.passes import ModulePass
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.common_subexpression_elimination import (
    CommonSubexpressionElimination,
)

from inconspiquous.transforms.xzs.convert_to_xzs import ConvertToXZS
from inconspiquous.transforms.xzs.lower import LowerXZSToSelect
from inconspiquous.transforms.xzs.fusion import XZSFusion
from inconspiquous.transforms.xzs.select import XZSSelect


class XZSSimpl(ModulePass):
    name = "xzs-simpl"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for p in (
            ConvertToXZS(),
            XZSSelect(),
            XZSFusion(),
            LowerXZSToSelect(),
            CommonSubexpressionElimination(),
            CanonicalizePass(),
        ):
            p.apply(ctx, op)
