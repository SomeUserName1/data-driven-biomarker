import random

import pandas as pd
import shap
import sklearn

from ddbm import FeatureAnalyzer

# a classic housing price dataset
X,y = shap.datasets.california(n_points=1000)

# a simple linear model
mmodel = sklearn.linear_model.LinearRegression()
mmodel.fit(X, y)

X['avg_house_value'] = y
df = X

sample_idxs = []
for i in range(0, 100):
    n = random.randint(0, df.shape[0])
    sample_idxs.append(n)

feature_analyzer = FeatureAnalyzer(df, sample_idxs, mmodel, mmodel.predict,
                                   list(df)[:-1], list(df)[-1])
print("=== SHAP Plots ===")
feature_analyzer.partial_dependence_plot(list(df)[1], 1)
feature_analyzer.scatter_plot(list(df)[1])
feature_analyzer.bar_plot()
feature_analyzer.waterfall_plot(1)
feature_analyzer.beeswarm_plot()
feature_analyzer.heatmap_plot()

print("=== Feature Cov matrix ===")
feature_analyzer.correlation_matrix_plot()

print("=== Zero Variance Features Removed ===")
feature_analyzer.remove_zero_variance_feats()

print("=== Shap Selection ===")
feature_analyzer.shap_select()

print("=== Forward Selection ===")
feature_analyzer.forwards_select()

print("=== Backwards Selection ===")
feature_analyzer.backwards_select()

print("=== Lasso-based Recursive Feature Elimintation===")
feature_analyzer.lasso_rfe_select()

print("done")
