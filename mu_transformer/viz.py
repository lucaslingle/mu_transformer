import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")

# # Load an example dataset with long-form data
# df = sns.load_dataset("fmri")

df = pd.read_csv("~/PycharmProjects/mu_transformer/temp.csv")
df["log_lr_base"] = df["lr_base"].map(lambda y: int(y.split("**")[-1].split(" ")[-1]))
print(df.head())

# Plot the responses for different events and regions
g = sns.FacetGrid(df, col="Parameterization", hue="d_model", legend_out=True)
g.map(sns.lineplot, "log_lr_base", "Val loss final", legend="brief", errorbar="se")
g.add_legend(title="width")
g.set_xlabels("log2(lr)")
g.set_ylabels("val loss")
# g.set(yticks=np.arange(2, 6, 1.0))
# g.set(xticks=np.arange(-12, 0, 2))
# plt.ylim(2, 6)
# plt.xlim(-12, -4)
plt.show()
