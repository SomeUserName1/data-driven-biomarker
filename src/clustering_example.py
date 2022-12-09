import random

import pandas as pd
import shap
import sklearn
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

from ddbm import FeatureAnalyzer

# a classic housing price dataset
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(
    n_samples=750, centers=centers, cluster_std=0.4, random_state=0
)

X = StandardScaler().fit_transform(X)

# a simple linear model
mmodel = DBSCAN(eps=0.3, min_samples=10)
mmodel.fit(X)


X['labels'] = labels_true 
df = X

sample_idxs = []
for i in range(0, 100):
    n = random.randint(0, df.shape[0])
    sample_idxs.append(n)

feature_analyzer = FeatureAnalyzer(df, sample_idxs, mmodel, None,
                                   list(df)[:-1], list(df)[-1], ModelType.CLUSTER)
feature_analyzer.partial_dependence_plot(list(df)[1], 1)
feature_analyzer.scatter_plot(list(df)[1])
feature_analyzer.bar_plot()
feature_analyzer.waterfall_plot(1)
feature_analyzer.beeswarm_plot()
feature_analyzer.heatmap_plot()
feature_analyzer.decision_plot()
feature_analyzer.force_plot()

print("done")
