import argparse

import seaborn as sns
import pandas as pd

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "input_file", help="path to csv file to read results from", type=str
)
input_file = arg_parser.parse_args().input_file

# Set styling
sns.set_theme(style="whitegrid")
sns.set_palette(("#1f78b4", "#d95f02", "#827b7b", "#a6cee3"))

# Load data
data = pd.read_csv(input_file)

# Create plot
ax = sns.lineplot(data, x="Instantiations", y="Runtime (s)", style="Method")
ax.set_xticks(tuple(100 * n for n in range(11)))
ax.set_xlim(xmin=0)
ax.set_yticks(tuple(range(14)))
ax.set_ylim(ymin=0)

# Save figure
fig = ax.get_figure()
fig.savefig("graph_rand_comp.pdf")
