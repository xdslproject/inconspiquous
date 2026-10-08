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

# Calculate cumulative values for each pass
for i in range(10):
    data.loc[i + 10, "Runtime (s)"] += data.loc[i, "Runtime (s)"]
    data.loc[i + 20, "Runtime (s)"] += data.loc[i + 10, "Runtime (s)"]
    data.loc[i + 30, "Runtime (s)"] += data.loc[i + 20, "Runtime (s)"]

line1 = data[data["Pass"] == "convert-to-xzs"]
line2 = data[data["Pass"] == "xzs-select"]
line3 = data[data["Pass"] == "xz-commute"]
line4 = data[data["Pass"] == "canonicalize"]

# Create plot
ax = sns.lineplot(data, x="Cycles", y="Runtime (s)", hue="Pass")
ax.set_xlim(xmin=0)
ax.set_ylim(ymin=0)
ax.fill_between(line1["Cycles"], 0, line1["Runtime (s)"])
ax.fill_between(line1["Cycles"], line1["Runtime (s)"], line2["Runtime (s)"])
ax.fill_between(line1["Cycles"], line2["Runtime (s)"], line3["Runtime (s)"])
ax.fill_between(line1["Cycles"], line3["Runtime (s)"], line4["Runtime (s)"])
ax.legend(title="Pass", prop={"family": "monospace"})

# Save figure
fig = ax.get_figure()
fig.savefig("graph_qec.pdf")
