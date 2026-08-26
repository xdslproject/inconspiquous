import argparse
import os.path
from pathlib import Path

from xdsl.context import Context
from xdsl.parser import Parser
from xdsl.transforms.canonicalize import CanonicalizePass

from inconspiquous.dialects import get_all_dialects
from inconspiquous.transforms.convert_qref_to_qssa import ConvertQrefToQssa
from inconspiquous.transforms.flip_coins import FlipCoinsPass
from inconspiquous.transforms.pauli_fusion import PauliFusionPass
from inconspiquous.transforms.randomized_comp import RandomizedComp

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "iterations", help="Number of random circuits to generate", type=int
)
iterations = arg_parser.parse_args().iterations

ghz_file = Path(
    os.path.dirname(__file__), "..", "tests", "filecheck", "examples", "ghz-10.mlir"
)

text = ghz_file.read_text()

ctx = Context()
for dialect_name, factory in get_all_dialects().items():
    ctx.register_dialect(dialect_name, factory)

mod = Parser(ctx, text, str(ghz_file)).parse_module()

for p in (
    ConvertQrefToQssa(),
    RandomizedComp(),
):
    p.apply(ctx, mod)

for i in range(0, iterations):
    print(i)
    c = mod.clone()

    for p in (
        FlipCoinsPass(),
        CanonicalizePass(),
        PauliFusionPass(),
    ):
        p.apply(ctx, c)
