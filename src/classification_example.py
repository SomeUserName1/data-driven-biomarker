import random

import pandas as pd
import shap
import sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from ddbm import FeatureAnalyzer, ModelType

# a classic housing price dataset
iris = load_iris()
X, y = iris.data, iris.target

cols = ['feature_' + str(i) for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=cols)
df['species'] = y

# a simple linear model
mmodel = sklearn.tree.DecisionTreeClassifier()
mmodel = mmodel.fit(df.iloc[:, :-1], df.iloc[:, -1])

sample_idxs = []
for i in range(0, 100):
    n = random.randint(0, df.shape[0] - 1)
    while(n in sample_idxs):
        n = random.randint(0, df.shape[0] - 1)

    sample_idxs.append(n)

feature_analyzer = FeatureAnalyzer(df, sample_idxs, mmodel, mmodel.predict,
                                   list(df)[:-1], list(df)[-1], ModelType.TREE)

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
