from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)

from inconspiquous.dialects.angle import ConstantAngleOp
from inconspiquous.dialects.measurement import (
    ConstantMeasurementOp,
    XYDynMeasurementOp,
    XYMeasurementAttr,
)


class XYDynMeasurementConst(RewritePattern):
    """
    Replaces a dynamic xy measurement with constant angle with a constant xy measurement.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: XYDynMeasurementOp, rewriter: PatternRewriter):
        if not isinstance(owner := op.angle.owner, ConstantAngleOp):
            return

        rewriter.replace_matched_op(
            ConstantMeasurementOp(XYMeasurementAttr(owner.angle))
        )
