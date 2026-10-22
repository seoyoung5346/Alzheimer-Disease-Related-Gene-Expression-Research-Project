import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv(
    "GSE85426_gene_level_expression.csv"
)

# 2. Separate AD and Control
ad_cols = [c for c in df.columns if c.endswith("_AD")]
ctrl_cols = [c for c in df.columns if c.endswith("_Ctrl")]

ad = df[ad_cols].astype(float)
ctrl = df[ctrl_cols].astype(float)

# 3. Analyze APP
app_index = df.index[df["Gene_Symbol"] == "APP"][0]

app_ad = ad.iloc[app_index]
app_ctrl = ctrl.iloc[app_index]

print("APP AD mean:", app_ad.mean())
print("APP Control mean:", app_ctrl.mean())
print("Mean difference:", app_ad.mean() - app_ctrl.mean())

print(ttest_ind(app_ad, app_ctrl, equal_var=False))

# 4. Draw box plot
plt.boxplot([app_ctrl, app_ad],
            tick_labels=["Control", "AD"])
plt.title("APP Expression: AD vs Control")
plt.ylabel("Expression Value")
plt.show()

# 5. Test all genes
test = ttest_ind(
    ad.to_numpy(),
    ctrl.to_numpy(),
    axis=1,
    equal_var=False
)

# 6. BH adjustment
adjusted_p = multipletests(
    test.pvalue,
    method="fdr_bh"
)[1]

# 7. Results
results = pd.DataFrame({
    "Gene_Symbol": df["Gene_Symbol"],
    "AD_mean": ad.mean(axis=1),
    "Control_mean": ctrl.mean(axis=1),
    "Mean_Difference": ad.mean(axis=1) - ctrl.mean(axis=1),
    "p_value": test.pvalue,
    "BH_adjusted_p": adjusted_p
})

results = results.sort_values("p_value")

print(results.head(10))
print("Raw significant:", (results.p_value < 0.05).sum())
print("BH significant:", (results.BH_adjusted_p < 0.05).sum())

results.to_csv(
    "GSE85426_gene_ttest_BH_results.csv",
    index=False
)
