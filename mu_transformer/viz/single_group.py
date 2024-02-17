import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--width")
args = parser.parse_args()


sns.set_theme(style="darkgrid")
sns.set(font_scale=0.5)

df = pd.read_csv("foo.csv", sep=",", header=0)
df = df[df["Width"] == int(args.width)]
df["log2_LR"] = df["LR"].map(lambda y: math.log2(y))
print(df.head())

# Plot the responses for different events and regions
g = sns.FacetGrid(
    data=df,
    col="Rule",
    hue="Group",
    # palette="viridis",
    col_order=["sp", "mup", "spectral"],
    height=2,
    aspect=1.0,
    gridspec_kws={"wspace": 0.15, "hspace": 0.10},
)
g.tight_layout()

g.figure.subplots_adjust(top=0.9)
g.set_titles("{col_name}")
g.map(sns.lineplot, "log2_LR", "Loss", legend="brief", linewidth=1.0)
g.add_legend(title="width")
g.figure.subplots_adjust(left=0.1)  # also try 0.05 for paper spacing it may look better
g.figure.subplots_adjust(bottom=0.2)
g.set_xlabels("log2(lr)")
g.set_ylabels("val loss")
g.set(xticks=np.arange(-10, 1, 2))
g.set(yticks=np.arange(2, 7, 1))
g.tick_params(axis="both", pad=0)
plt.ylim(2, 6)
plt.xlim(-10, 0)
plt.savefig(f"{args.width}.pdf", format="pdf")
