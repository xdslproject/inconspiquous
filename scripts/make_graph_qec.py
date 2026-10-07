# Import seaborn
import seaborn as sns
import pandas as pd

# Apply the default theme
sns.set_theme(style="whitegrid")

sns.set_palette(("#1f78b4", "#d95f02", "#827b7b", "#a6cee3"))  # , '#b2df8a'))

# Load an example dataset
data = pd.read_csv("results_qec.csv")

for i in range(10):
    data.loc[i + 10, "Runtime (s)"] += data.loc[i, "Runtime (s)"]
    data.loc[i + 20, "Runtime (s)"] += data.loc[i + 10, "Runtime (s)"]
    data.loc[i + 30, "Runtime (s)"] += data.loc[i + 20, "Runtime (s)"]
    # data.loc[i + 40, "Runtime (s)"] += data.loc[i + 30, "Runtime (s)"]

line1 = data[data["Pass"] == "convert-to-xzs"]
line2 = data[data["Pass"] == "xzs-select"]
line3 = data[data["Pass"] == "xz-commute"]
line4 = data[data["Pass"] == "canonicalize"]
# line5 = data[data["Pass"]=="cse"]

print(line1)
print(line2)
print(line3)
print(line4)
# print(line5)

ax = sns.lineplot(data, x="Cycles", y="Runtime (s)", hue="Pass")

# ax.set_xscale("log")
# ax.set_xticks(tuple(100*n for n in range(11)))
ax.set_xlim(xmin=0)
# ax.set_yscale("log")
# ax.set_yticks(tuple(range(14)))
ax.set_ylim(ymin=0)
ax.fill_between(line1["Cycles"], 0, line1["Runtime (s)"])
ax.fill_between(line1["Cycles"], line1["Runtime (s)"], line2["Runtime (s)"])
ax.fill_between(line1["Cycles"], line2["Runtime (s)"], line3["Runtime (s)"])
ax.fill_between(line1["Cycles"], line3["Runtime (s)"], line4["Runtime (s)"])
# ax.fill_between(line1["Cycles"], line4["Runtime (s)"], line5["Runtime (s)"])
ax.legend(title="Pass", prop={"family": "monospace"})

fig = ax.get_figure()
fig.savefig("graph_qec.pdf")
#
