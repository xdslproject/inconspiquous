from inconspiquous.dialects import get_all_dialects
from inconspiquous.transforms import get_all_passes
from xdsl.xdsl_opt_main import xDSLOptMain


class QuoptMain(xDSLOptMain):
    def register_all_dialects(self):
        for name, dialect in get_all_dialects().items():
            self.ctx.register_dialect(name, dialect)

    def register_all_passes(self):
        for name, pass_ in get_all_passes().items():
            self.register_pass(name, pass_)

    def register_all_targets(self):
        super().register_all_targets()
        try:
            from inconspiquous.backend.qir import print_qir

            self.available_targets["qir"] = print_qir
        except ImportError:
            # PyQIR not available
            pass


def main():
    quopt_main = QuoptMain()
    quopt_main.run()


if "__main__" == __name__:
    main()
