import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")

df = pd.read_csv("foo.csv", sep=",", header=0)
df["log2_LR"] = df["LR"].map(lambda y: math.log2(y))
print(df.head())

# Plot the responses for different events and regions
g = sns.FacetGrid(
    data=df,
    col="Rule",
    hue="Width",
    palette="viridis",
    col_order=["sp", "mup", "spectral"],
    gridspec_kws={"wspace": 0.15},
    height=2,
    aspect=1,
)
g.set_titles("{col_name}")
g.map(sns.lineplot, "log2_LR", "Loss", legend="brief")
g.add_legend(title="width")
g.set_xlabels("log2(lr)")
g.set_ylabels("val loss")
g.set(xticks=np.arange(-10, 1, 2))
g.set(yticks=np.arange(2, 7, 1))
plt.ylim(2, 6)
plt.xlim(-10, 0)
plt.show()
