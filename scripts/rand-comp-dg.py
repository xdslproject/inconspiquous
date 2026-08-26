import argparse
import os.path
from pathlib import Path

from xdsl.context import Context
from xdsl.parser import Parser
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.common_subexpression_elimination import (
    CommonSubexpressionElimination,
)

from inconspiquous.dialects import get_all_dialects
from inconspiquous.transforms.convert_qref_to_qssa import ConvertQrefToQssa
from inconspiquous.transforms.flip_coins import FlipCoinsPass
from inconspiquous.transforms.randomized_comp import RandomizedComp
from inconspiquous.transforms.xzs.convert_to_xzs import ConvertToXZS
from inconspiquous.transforms.xzs.fusion import XZSFusion
from inconspiquous.transforms.xzs.lower import LowerXZSToSelect
from inconspiquous.transforms.xzs.select import XZSSelect

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
    ConvertToXZS(),
    XZSSelect(),
    XZSFusion(),
    CommonSubexpressionElimination(),
    CanonicalizePass(),
    LowerXZSToSelect(),
    CommonSubexpressionElimination(),
):
    p.apply(ctx, mod)

for i in range(0, iterations):
    print(i)
    c = mod.clone()

    for p in (
        FlipCoinsPass(),
        CanonicalizePass(),
    ):
        p.apply(ctx, c)
