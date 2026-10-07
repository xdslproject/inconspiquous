# Import seaborn
import seaborn as sns
from scipy.stats import gmean
import pandas as pd

from matplotlib import rcParams

# figure size in inches
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
    if x.iloc[0] == "Dynamic gate":
        return "Dynamic gate"
    elif x.iloc[0] == "QIR":
        return "QIR"
    try:
        return gmean(x)
    except:
        return "Geomean"


def plot(file_name, column):
    # Apply the default theme
    sns.set_theme(style="whitegrid")

    sns.set_palette(("#1f78b4", "#d95f02", "#827b7b", "#a6cee3"))

    data = pd.read_csv("results_benchmark.csv")
    data.loc[len(data.index)] = data[data["IR"] == "Dynamic gate"].aggregate(my_gmean)
    data.loc[len(data.index)] = data[data["IR"] == "QIR"].aggregate(my_gmean)
    print(data)

    print(f"Plotting {column}")
    ax = sns.barplot(data, x="Benchmark", y=column, hue="IR")

    ax.tick_params("x", labelrotation=50, labelsize=15)
    ax.tick_params("y", labelsize=15)
    ax.legend(ncol=2, facecolor="white", fontsize=17)
    ax.xaxis.label.set_text("")
    ax.xaxis.label.set_fontsize(0)
    ax.yaxis.label.set_fontsize(17)

    fig = ax.get_figure()

    print(f"Saving to graph_{file_name}.pdf")
    fig.savefig(f"graph_{file_name}.pdf", bbox_inches="tight")
    fig.clear()


for file_name, column in COLUMNS:
    plot(file_name, column)
