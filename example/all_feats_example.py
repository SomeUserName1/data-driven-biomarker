import random
import math
from pprint import pprint
import pickle
import sys
import multiprocessing
sys.path.append(r"/home/someusername/sync/workspace/nb_tue/3/2_gharabaghi/data-driven-biomarker/src/")
sys.path.append(r"/home/someusername/sync/workspace/nb_tue/3/2_gharabaghi/intraop-neuro-phys-dataset_private")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import sklearn
import xgboost
from ddbm import FeatureAnalyzer, ModelType
#from Group import IntraOpNeuroDataset

pkl_path = "/home/someusername/sync/workspace/nb_tue/3/2_gharabaghi/group_1669237550.pkl"

with open(pkl_path, "rb") as group_file:
    intraop_data = pickle.load(group_file)

# each row one contact
df_lfp = intraop_data.lfp_features_df[['aligned depth', 'lfp psd']]
# each row one eeg
# pandas can do it: pd.explode

df = df_lfp['lfp psd'].apply(lambda x: pd.Series(np.mean(x, axis=0)))

group_sz = 3
binned_df = pd.DataFrame()
col_segments = range(math.ceil(df.shape[1] / float(group_sz)))
for col in col_segments:
    binned_df[str(col * group_sz) + "_" + str((col + 1) * group_sz - 1) + '_Hz'] = df.iloc[:, group_sz * col : group_sz * (col + 1)].mean(axis=1)

del df
df = binned_df
# df['in_stn'] = [1 if d <= 1.75 and d >= -4 else -1 for d in df_lfp['aligned depth']]

#mmodel = sklearn.tree.DecisionTreeClassifier(max_depth=5)
#mmodel.fit(df.iloc[:, :-1], df.iloc[:, -1])
#sklearn.tree.plot_tree(mmodel)
#plt.show()
#print(mmodel.feature_importances_)
#
#sample_idxs = df[df['in_stn'] == 1].index.to_list()
#for i in range(0, 333):
#    n = random.randint(0, df.shape[0] - 1)
#    while n in sample_idxs:
#        n = random.randint(0, df.shape[0] - 1)
#
#    sample_idxs.append(n)
#
#feature_analyzer = FeatureAnalyzer(df, sample_idxs, mmodel, mmodel.predict,
#                                   list(df)[:-1], list(df)[-1], ModelType.TREE)
#
#print("=== SHAP Plots ===")
#feature_analyzer.bar_plot()
#feature_analyzer.beeswarm_plot()
#feature_analyzer.heatmap_plot()
#feature_analyzer.decision_plot()
#
#print("=== Feature Cov matrix ===")
#feature_analyzer.correlation_matrix_plot()
#
#print("=== Zero Variance Features Removed ===")
#feature_analyzer.remove_zero_variance_feats()
#
#print("=== Shap Selection ===")
#feature_analyzer.shap_select()
#
#print("=== Forward Selection ===")
#feature_analyzer.forwards_select()
#
#print("=== Backwards Selection ===")
#feature_analyzer.backwards_select()
#
#print("=== Logit-based Recursive Feature Elimintation===")
#feature_analyzer.logit_rfe_select()

df['depth'] = df_lfp['aligned depth']

mmodel = xgboost.XGBRegressor(n_jobs=multiprocessing.cpu_count() // 2, max_depth=5)
mmodel.fit(df.iloc[:, :-1], df.iloc[:, -1])
xgboost.plot_importance(mmodel)
xgboost.plot_tree(mmodel)

sample_idxs = []
for idx, d in enumerate(df_lfp['aligned depth']):
    if d <= 1.75 and d >= -4 :
        sample_idxs.append(idx)

for i in range(0, 333):
    n = random.randint(0, df.shape[0] - 1)
    while n in sample_idxs:
        n = random.randint(0, df.shape[0] - 1)

    sample_idxs.append(n)

feature_analyzer = FeatureAnalyzer(df, sample_idxs, mmodel, mmodel.predict,
                                   list(df)[:-1], list(df)[-1], ModelType.TREE)

print("=== SHAP Plots ===")
feature_analyzer.bar_plot()
feature_analyzer.beeswarm_plot()
feature_analyzer.heatmap_plot()
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

print("=== Logit-based Recursive Feature Elimintation===")
feature_analyzer.lasso_rfe_select()

