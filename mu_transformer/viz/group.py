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

df = pd.read_csv("main_results.csv", sep="\t", header=0)
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
# ax.set_title(args.group, fontsize=10)
ax.set_xlabel("log2(α)", fontsize=10)
ax.set_ylabel("Loss", fontsize=10)
ax.set(xticks=np.arange(-10, -1, 2))
ax.set(yticks=np.arange(2, 6, 1))

ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=10)

ax.legend(title="Width", fontsize=10, title_fontsize=10)

plt.ylim(2, 5)
plt.xlim(-10, -2)
plt.savefig(f"outputs/{args.group}.pdf", format="pdf")
