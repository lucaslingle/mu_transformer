import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")

df = pd.read_csv("temp_dev19.csv", sep="	", header=0)
df["log2_LR"] = df["LR"].map(lambda y: math.log2(y))
print(df.head())

# Plot the responses for different events and regions
g = sns.FacetGrid(df, col="Rule", hue="Width", palette="viridis")
g.set_titles("{col_name}")
g.map(sns.lineplot, "log2_LR", "Loss", legend="brief")
g.add_legend(title="width")
g.set_xlabels("log2(lr)")
g.set_ylabels("val loss")
# g.set(yticks=np.arange(2, 6, 1.0))
# g.set(xticks=np.arange(-12, 0, 2))
plt.ylim(2, 4)
plt.xlim(-12, -1)
plt.show()
