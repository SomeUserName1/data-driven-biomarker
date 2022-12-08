import random

import pandas as pd
import shap
import sklearn

from ddbm import FeatureAnalyzer

# a classic housing price dataset
X,y = shap.datasets.california(n_points=1000)

# a simple linear model
model = sklearn.linear_model.LinearRegression()
model.fit(X, y)

df = X['avg_house_value'] = y

sample_idxs = []
for i in range(0, 100):
    n = random.randint(0, len(df.shape[0]))
    sample_idxs.append(n)

feature_analyzer = FeatureAnalyzer(df, sample_idxs, model, model.predict,
                                   list(df)[:-1], list(df)[-1])
feature_analyzer.partial_dependence_plot(list(df)[1], 1)
feature_analyzer.scatter_plot(list(df)[1])
feature_analyzer.bar_plot()
feature_analyzer.waterfall_plot(1)
feature_analyzer.beeswarm_plot()
feature_analyzer.heatmap_plot()
feature_analyzer.decision_plot()
feature_analyzer.force_plot()
