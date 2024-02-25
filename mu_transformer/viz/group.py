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

df = pd.read_csv("main_results.csv", sep="	", header=0)
df = df[df["Group"] == args.group]
df["log2_LR"] = df["LR"].map(lambda y: math.log2(y))
print(df.head())

ax = sns.lineplot(
    data=df,
    x="log2_LR",
    y="Loss",
    hue="Width",
    palette=sns.color_palette("viridis", n_colors=3),
)
ax.set_title(args.group, fontsize=10)
ax.set_xlabel("log2(η)", fontsize=8)
ax.set_ylabel("Validation Loss", fontsize=8)
ax.set(xticks=np.arange(-10, 1, 2))
ax.set(yticks=np.arange(2, 5, 0.5))

ax.tick_params(axis="x", labelsize=8)
ax.tick_params(axis="y", labelsize=8)

ax.legend(fontsize=8)

plt.ylim(2, 4)
plt.xlim(-10, 0)
plt.savefig(f"{args.group}.pdf", format="pdf")
