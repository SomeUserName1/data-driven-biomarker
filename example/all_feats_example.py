import random
import pickle
import sys
sys.path.append(r"/home/someusername/sync/workspace/nb_tue/3/2_gharabaghi/data-driven-biomarker/src/")
sys.path.append(r"/home/someusername/sync/workspace/nb_tue/3/2_gharabaghi/intraop-neuro-phys-dataset_private")

import pandas as pd
import shap
import sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from ddbm import FeatureAnalyzer, ModelType

pkl_path = "/home/someusername/sync/workspace/nb_tue/3/2_gharabaghi/group_1669237550.pkl"

group_file = open(pkl_path, "rb")
intraop_data = pickle.load(group_file)
group_file.close()

print(intraop_data)
# each row one contact
df_lfp = intraop_data.lfp_features_df
# each row one eeg
df_eeg = intraop_data.eeg_features_df
# pandas can do it 


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




