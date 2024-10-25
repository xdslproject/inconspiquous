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

    return {
        "canonicalize": get_canonicalize,
        "convert-qssa-to-qref": get_convert_qssa_to_qref,
        "convert-scf-to-cf": get_convert_scf_to_cf,
    }
