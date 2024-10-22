from collections.abc import Callable

from xdsl.passes import ModulePass


def get_all_passes() -> dict[str, Callable[[], type[ModulePass]]]:
    """Returns all available passes."""

    def get_convert_qssa_to_qref():
        from inconspiquous.transforms import convert_qssa_to_qref

        return convert_qssa_to_qref.ConvertQssaToQref

    return {
        "convert-qssa-to-qref": get_convert_qssa_to_qref,
    }
