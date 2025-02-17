from typing import cast
from xdsl.builder import ImplicitBuilder
from xdsl.dialects import arith, scf, varith
from xdsl.dialects import builtin
from xdsl.ir import Block, Region, SSAValue
from xdsl.dialects.builtin import IndexType
from xdsl.parser import DenseArrayBase, IntegerType, Context
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)

from inconspiquous.dialects.qssa import DynGateOp


class LowerDynGateToScfPattern(RewritePattern):
    """
    Attempts to lower a qssa.dyn_gate op to scf
    When the gate argument is an arith.select we replace the gate with an scf.if
    When the gate argument is a varith.switch we replace the gate with an scf.index_switch
    """

    @staticmethod
    def make_region_from_arg(
        op: DynGateOp, gate: SSAValue, rewriter: PatternRewriter
    ) -> Region:
        region = Region(Block())
        with ImplicitBuilder(region):
            dyn_gate = DynGateOp(gate, *op.ins)
            scf.YieldOp(dyn_gate)
        return region

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DynGateOp, rewriter: PatternRewriter):
        gate = op.gate.owner
        if isinstance(gate, arith.SelectOp):
            then_region = self.make_region_from_arg(op, gate.lhs, rewriter)
            else_region = self.make_region_from_arg(op, gate.rhs, rewriter)
            rewriter.replace_matched_op(
                scf.IfOp(gate.cond, op.result_types, then_region, else_region)
            )
        elif isinstance(gate, varith.VarithSwitchOp):
            flag = arith.IndexCastOp(gate.flag, IndexType())
            cases = DenseArrayBase.from_list(
                IntegerType(64),
                tuple(cast(int, x) for x in gate.case_values.iter_values()),
            )

            default_region = self.make_region_from_arg(op, gate.default_arg, rewriter)
            case_regions = tuple(
                self.make_region_from_arg(op, x, rewriter) for x in gate.args
            )

            rewriter.replace_matched_op(
                (
                    flag,
                    scf.IndexSwitchOp(
                        flag, cases, default_region, case_regions, op.result_types
                    ),
                )
            )


class LowerDynGateToScf(ModulePass):
    name = "lower-dyn-gate-to-scf"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(LowerDynGateToScfPattern()).rewrite_module(op)
