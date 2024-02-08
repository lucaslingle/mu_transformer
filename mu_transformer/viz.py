import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")

df = pd.read_csv("temp.csv")
df["log_lr_base"] = df["lr_base"].map(lambda y: int(y.split("**")[-1].split(" ")[-1]))
print(df.head())

# Plot the responses for different events and regions
g = sns.FacetGrid(df, col="Parameterization", hue="d_model", palette="viridis")
g.set_titles("{col_name}")
g.map(sns.lineplot, "log_lr_base", "Val loss final", legend="brief")
g.add_legend(title="width")
g.set_xlabels("log2(lr)")
g.set_ylabels("val loss")
# g.set(yticks=np.arange(2, 6, 1.0))
# g.set(xticks=np.arange(-12, 0, 2))
# plt.ylim(2, 6)
# plt.xlim(-12, -4)
plt.show()
