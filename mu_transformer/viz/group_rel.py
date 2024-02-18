import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--group")
args = parser.parse_args()


sns.set_theme(style="darkgrid")
sns.set(font_scale=0.5)

df = pd.read_csv("main_results.csv", sep=",", header=0)
df = df[df["Group"] == args.group]
df["log2_LR"] = df["LR"].map(lambda y: math.log2(y))
print(df.head())

# Plot the responses for different events and regions
g = sns.relplot(
    data=df,
    x="log2_LR",
    y="Loss",
    col="Rule",
    hue="Width",
    palette="viridis",
    col_order=["sp", "mup", "spectral"],
    height=2,
    aspect=1.0,
    kind="line",
    facet_kws={"gridspec_kws": {"wspace": 0.15, "hspace": 0.10}},
)

g.tight_layout()
g.figure.subplots_adjust(top=0.9)
g.set_titles("{col_name}")
g.figure.subplots_adjust(left=0.1)
g.figure.subplots_adjust(bottom=0.2)
g.set_xlabels("log2(lr)")
g.set_ylabels("val loss")
g.set(xticks=np.arange(-10, 1, 2))
g.set(yticks=np.arange(2, 7, 1))
g.tick_params(axis="both", pad=0)
plt.ylim(2, 6)
plt.xlim(-10, 0)
plt.savefig(f"{args.group}.pdf", format="pdf")
