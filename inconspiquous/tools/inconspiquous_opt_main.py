from xdsl.xdsl_opt_main import xDSLOptMain


class OptMain(xDSLOptMain):
    def register_all_dialects(self):
        super().register_all_dialects()
        ## Add custom dialects
        # FIXME: override upstream qref dialect. Remove this after upstreaming full downstream qref dialect.
        self.ctx._registered_dialects.pop("qref", None)  # pyright: ignore


def main():
    xdsl_main = OptMain()
    xdsl_main.run()


if "__main__" == __name__:
    main()
