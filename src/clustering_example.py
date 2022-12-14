import random

import pandas as pd
import shap
import numpy as np 
import scipy as sp
import sklearn
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

from ddbm import FeatureAnalyzer, ModelType

centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(
    n_samples=750, centers=centers, cluster_std=0.4, random_state=0
)

X = StandardScaler().fit_transform(X)
cols = ['feature_' + str(i) for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=cols)

# a simple linear model
mmodel = DBSCAN(eps=0.3, min_samples=10)
mmodel.fit(df)

df['labels'] = mmodel.labels_
n_clusters = len(set(df['labels']))
df['labels'] = df['labels'].mask(df['labels'] == -1, n_clusters - 1)

sample_idxs = []
for i in range(0, 100):
    n = random.randint(0, df.shape[0] - 1)
    while(n in sample_idxs):
        n = random.randint(0, df.shape[0] - 1)

    sample_idxs.append(n)

feature_analyzer = FeatureAnalyzer(df, sample_idxs, mmodel, None,
                                   list(df)[:-1], list(df)[-1], model_type=ModelType.CLUSTER)
print("=== SHAP Plots ===")
feature_analyzer.partial_dependence_plot(1)
feature_analyzer.scatter_plot(1)
feature_analyzer.bar_plot()
feature_analyzer.waterfall_plot(1)
feature_analyzer.beeswarm_plot()
feature_analyzer.heatmap_plot()
feature_analyzer.force_plot(1)
feature_analyzer.decision_plot()

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
