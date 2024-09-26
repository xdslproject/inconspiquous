from inconspiquous.dialects import get_all_dialects
from xdsl.xdsl_opt_main import xDSLOptMain


class QuoptMain(xDSLOptMain):
    def register_all_dialects(self):
        for name, dialect in get_all_dialects().items():
            self.ctx.register_dialect(name, dialect)

    def register_all_passes(self):
        super().register_all_passes()

    def register_all_targets(self):
        super().register_all_targets()


def main():
    quopt_main = QuoptMain()
    quopt_main.run()


if "__main__" == __name__:
    main()
