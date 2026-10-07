# Import seaborn
import seaborn as sns
import pandas as pd

# Apply the default theme
sns.set_theme(style="whitegrid")

sns.set_palette(("#1f78b4", "#d95f02", "#827b7b", "#a6cee3"))

# Load an example dataset
data = pd.read_csv("results_rand_comp.csv")

ax = sns.lineplot(data, x="Instantiations", y="Runtime (s)", style="Method")

# ax.set_xscale("log")
ax.set_xticks(tuple(100 * n for n in range(11)))
ax.set_xlim(xmin=0)
# ax.set_yscale("log")
ax.set_yticks(tuple(range(14)))
ax.set_ylim(ymin=0)
fig = ax.get_figure()
fig.savefig("graph_rand_comp.pdf")
