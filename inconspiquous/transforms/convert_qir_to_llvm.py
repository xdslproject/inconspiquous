from xdsl.ir import dataclass, field
from xdsl.parser import Context, ModuleOp
from xdsl.passes import ModulePass
from xdsl.dialects import llvm
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    TypeConversionPattern,
    attr_type_rewrite_pattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint

from inconspiquous.dialects.qir import QIROperation, QubitType, ResultType


class QIRTypeToLLVMPointerPattern(TypeConversionPattern):
    @attr_type_rewrite_pattern
    def convert_type(self, typ: QubitType | ResultType) -> llvm.LLVMPointerType:
        return llvm.LLVMPointerType()


@dataclass
class QIRToLLVMPattern(RewritePattern):
    """
    Convert a qir operation to its llvm opaque function.
    Declares the llvm function the first time an operation is encountered.
    """

    module: ModuleOp

    declared_functions: set[str] = field(default_factory=set[str])

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: QIROperation, rewriter: PatternRewriter):
        func_name = op.get_func_name()
        func_type = op.get_func_type()

        if func_name not in self.declared_functions:
            rewriter.insert_op(
                llvm.FuncOp(func_name, func_type, linkage=llvm.LinkageAttr("external")),
                InsertPoint.at_start(self.module.body.block),
            )
            self.declared_functions.add(func_name)

        output = func_type.output

        rewriter.replace_matched_op(
            llvm.CallOp(
                func_name,
                *op.operands,
                return_type=None if output == llvm.LLVMVoidType() else output,
            )
        )


class ConvertQIRTLLVMPass(ModulePass):
    name = "convert-qir-to-llvm"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [QIRTypeToLLVMPointerPattern(recursive=True), QIRToLLVMPattern(op)]
            )
        ).rewrite_module(op)
