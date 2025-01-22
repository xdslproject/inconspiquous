from collections.abc import Callable

from xdsl.passes import ModulePass


def get_all_passes() -> dict[str, Callable[[], type[ModulePass]]]:
    """Returns all available passes."""

    def get_canonicalize():
        from xdsl.transforms import canonicalize

        return canonicalize.CanonicalizePass

    def get_convert_qref_to_qssa():
        from inconspiquous.transforms import convert_qref_to_qssa

        return convert_qref_to_qssa.ConvertQrefToQssa

    def get_convert_qssa_to_qref():
        from inconspiquous.transforms import convert_qssa_to_qref

        return convert_qssa_to_qref.ConvertQssaToQref

    def get_convert_scf_to_cf():
        from xdsl.transforms import convert_scf_to_cf

        return convert_scf_to_cf.ConvertScfToCf

    def get_convert_to_xzs():
        from inconspiquous.transforms.xzs import convert_to_xzs

        return convert_to_xzs.ConvertToXZS

    def get_cse():
        from xdsl.transforms import common_subexpression_elimination

        return common_subexpression_elimination.CommonSubexpressionElimination

    def get_dce():
        from xdsl.transforms import dead_code_elimination

        return dead_code_elimination.DeadCodeElimination

    def get_lower_dyn_gate_to_scf():
        from inconspiquous.transforms import lower_dyn_gate_to_scf

        return lower_dyn_gate_to_scf.LowerDynGateToScf

    def get_lower_to_fin_supp():
        from inconspiquous.transforms import lower_to_fin_supp

        return lower_to_fin_supp.LowerToFinSupp

    def get_lower_xzs_to_select():
        from inconspiquous.transforms.xzs import lower

        return lower.LowerXZSToSelect

    def get_mlir_opt():
        from xdsl.transforms import mlir_opt

        return mlir_opt.MLIROptPass

    def get_randomized_comp():
        from inconspiquous.transforms import randomized_comp

        return randomized_comp.RandomizedComp

    def get_qec_inline():
        from inconspiquous.transforms.qec import inline

        return inline.QECInlinerPass

    def get_xz_commute():
        from inconspiquous.transforms.xzs import commute

        return commute.XZCommute

    def get_xzs_merge():
        from inconspiquous.transforms.xzs import merge

        return merge.XZSMerge

    def get_xzs_select():
        from inconspiquous.transforms.xzs import select

        return select.XZSSelect

    def get_xzs_simpl():
        from inconspiquous.transforms.xzs import pipeline

        return pipeline.XZSSimpl

    return {
        "canonicalize": get_canonicalize,
        "convert-qref-to-qssa": get_convert_qref_to_qssa,
        "convert-qssa-to-qref": get_convert_qssa_to_qref,
        "convert-scf-to-cf": get_convert_scf_to_cf,
        "convert-to-xzs": get_convert_to_xzs,
        "cse": get_cse,
        "dce": get_dce,
        "qec-inline": get_qec_inline,
        "lower-dyn-gate-to-scf": get_lower_dyn_gate_to_scf,
        "lower-to-fin-supp": get_lower_to_fin_supp,
        "lower-xzs-to-select": get_lower_xzs_to_select,
        "mlir-opt": get_mlir_opt,
        "randomized-comp": get_randomized_comp,
        "xz-commute": get_xz_commute,
        "xzs-merge": get_xzs_merge,
        "xzs-select": get_xzs_select,
        "xzs-simpl": get_xzs_simpl,
    }
