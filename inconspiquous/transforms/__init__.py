from collections.abc import Callable

from xdsl.passes import ModulePass


def get_all_passes() -> dict[str, Callable[[], type[ModulePass]]]:
    """Returns all available passes."""

    def get_canonicalize():
        from xdsl.transforms import canonicalize

        return canonicalize.CanonicalizePass

    def get_convert_qssa_to_qref():
        from inconspiquous.transforms import convert_qssa_to_qref

        return convert_qssa_to_qref.ConvertQssaToQref

    def get_convert_scf_to_cf():
        from xdsl.transforms import convert_scf_to_cf

        return convert_scf_to_cf.ConvertScfToCf

    def get_convert_to_xs():
        from inconspiquous.transforms.xs import convert_to_xs

        return convert_to_xs.ConvertToXS

    def get_cse():
        from xdsl.transforms import common_subexpression_elimination

        return common_subexpression_elimination.CommonSubexpressionElimination

    def get_dce():
        from xdsl.transforms import dead_code_elimination

        return dead_code_elimination.DeadCodeElimination

    def get_lower_dyn_gate_to_scf():
        from inconspiquous.transforms import lower_dyn_gate_to_scf

        return lower_dyn_gate_to_scf.LowerDynGateToScf

    def get_lower_xs_to_select():
        from inconspiquous.transforms.xs import lower

        return lower.LowerXSToSelect

    def get_merge_xs():
        from inconspiquous.transforms.xs import merge

        return merge.MergeXSGates

    def get_mlir_opt():
        from xdsl.transforms import mlir_opt

        return mlir_opt.MLIROptPass

    def get_randomized_comp():
        from inconspiquous.transforms import randomized_comp

        return randomized_comp.RandomizedComp

    def get_xs_select():
        from inconspiquous.transforms.xs import select

        return select.XSSelect

    return {
        "canonicalize": get_canonicalize,
        "convert-qssa-to-qref": get_convert_qssa_to_qref,
        "convert-scf-to-cf": get_convert_scf_to_cf,
        "convert-to-xs": get_convert_to_xs,
        "cse": get_cse,
        "dce": get_dce,
        "lower-dyn-gate-to-scf": get_lower_dyn_gate_to_scf,
        "lower-xs-to-select": get_lower_xs_to_select,
        "merge-xs": get_merge_xs,
        "mlir-opt": get_mlir_opt,
        "randomized-comp": get_randomized_comp,
        "xs-select": get_xs_select,
    }
