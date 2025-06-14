from __future__ import annotations

from typing import IO
from xdsl.xdsl_opt_main import xDSLOptMain
from xdsl.ir import ModuleOp

from inconspiquous.backend.qir import pyqir_available


class QuoptMain(xDSLOptMain):
    """Main entry point for quantum optimization passes"""

    def register_all_dialects(self):
        """Register all dialects"""
        super().register_all_dialects()
        # Add quantum-specific dialects here

    def register_all_passes(self):
        """Register all passes"""
        super().register_all_passes()
        # Add quantum-specific passes here

    def register_all_targets(self):
        super().register_all_targets()
        try:
            from inconspiquous.backend.qir import emit_qir_module

            def print_qir(module: ModuleOp, output: IO[str]) -> None:
                try:
                    qir_module = emit_qir_module(module)
                    output.write(str(qir_module.ir()))
                except Exception as e:
                    output.write(f"; Error generating QIR: {e}\n")
                    output.write("; Module structure:\n")
                    for op in module.walk():
                        if hasattr(op, "name"):
                            output.write(f";   {op.name}\n")

            self.available_targets["qir"] = print_qir
        except ImportError:
            # PyQIR not available
            pass

    def run_passes(self, module: ModuleOp) -> None:
        """Run passes on module"""
        super().run_passes(module)

        # If PyQIR is available, emit QIR
        if pyqir_available:
            from inconspiquous.backend.qir import emit_qir_module

            emit_qir_module(module)

        # Walk through all operations and print their names
        for op in module.walk():
            if hasattr(op, "name"):
                print(f"Operation: {op.name}")


def main():
    """Main entry point"""
    QuoptMain().run()


if __name__ == "__main__":
    main()
