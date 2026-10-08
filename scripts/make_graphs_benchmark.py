import argparse

import seaborn as sns
from scipy.stats import gmean
import pandas as pd

from matplotlib import rcParams


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "input_file", help="path to csv file to read results from", type=str
)
input_file = arg_parser.parse_args().input_file

# Set styling
sns.set_theme(style="whitegrid")
sns.set_palette(("#1f78b4", "#d95f02", "#827b7b", "#a6cee3"))
rcParams["figure.figsize"] = [8.4, 4.5]


COLUMNS = [
    ("line_count", "Line count"),
    ("word_count", "Word count"),
    ("basic_blocks", "Basic blocks"),
    ("cyclomatic", "Cyclomatic complexity"),
    ("quantum", "Quantum gates"),
    ("halstead", "Halstead difficulty"),
]

def my_gmean(x):
    """
    Calculates geomean of a numeric column, otherwise providing the correct string.
    """
    # Check if we are processing the "IR" column
    if x.iloc[0] == "Dynamic gate":
        return "Dynamic gate"
    elif x.iloc[0] == "QIR":
        return "QIR"
    try:
        # Otherwise try to take numeric geomean
        return gmean(x)
    except:
        # If this fails we were processing the "Benchmark column"
        # This is obviously hacky but works anyway.
        return "Geomean"

# Add geomean rows
data = pd.read_csv(input_file)
data.loc[len(data.index)] = data[data["IR"] == "Dynamic gate"].aggregate(my_gmean)
data.loc[len(data.index)] = data[data["IR"] == "QIR"].aggregate(my_gmean)

def plot(file_name, column):
    print(f"Plotting {column}")
    ax = sns.barplot(data, x="Benchmark", y=column, hue="IR")

    ax.tick_params("x", labelrotation=50, labelsize=15)
    ax.tick_params("y", labelsize=15)
    ax.legend(ncol=2, facecolor="white", fontsize=17)
    ax.xaxis.label.set_text("")
    ax.xaxis.label.set_fontsize(0)
    ax.yaxis.label.set_fontsize(17)

    # Save figure
    fig = ax.get_figure()

    print(f"Saving to graph_{file_name}.pdf")
    fig.savefig(f"graph_{file_name}.pdf", bbox_inches="tight")
    fig.clear()


for file_name, column in COLUMNS:
    plot(file_name, column)
