import random
import math
from pprint import pprint
import pickle
import sys
import multiprocessing
sys.path.append(r"/home/someusername/sync/workspace/nb_tue/3/2_gharabaghi/data-driven-biomarker/src/")
sys.path.append(r"/home/someusername/sync/workspace/nb_tue/3/2_gharabaghi/intraop-neuro-phys-dataset_private")

from sklearn.manifold import TSNE
from umap import UMAP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import sklearn
import xgboost
from ddbm import FeatureAnalyzer, ModelType
from sklearn.cluster import OPTICS
import hdbscan
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

############### CLUSTERING of PSDs per depth #######################################################
df['depth'] = df_lfp['aligned depth']

mmodel = hdbscan.HDBSCAN(min_samples=3, min_cluster_size=100, cluster_selection_method='eom')
mmodel.fit(df.iloc[:, :-1])

df['cluster'] = mmodel.labels_
n_clus = mmodel.labels_.max() + 1
df['cluster'] = df['cluster'].mask(df['cluster'] == -1, n_clus)

color_palette = sns.color_palette('Paired', n_clus +1 )
cluster_colors = [color_palette[x] if x >= 0
                  else (0.5, 0.5, 0.5)
                  for x in mmodel.labels_]
cluster_member_colors = [sns.desaturate(x, p) for x, p in
                         zip(cluster_colors, mmodel.probabilities_)]
data = UMAP(n_components=3).fit_transform(df.iloc[:, :-2])
ax = plt.figure().add_subplot(projection='3d')
ax.scatter(xs=data.T[0], ys=data.T[1], zs=data.T[2], linewidth=0, c=cluster_member_colors, alpha=0.25)
plt.show()

mmodel.condensed_tree_.plot(select_clusters=True,
                               selection_palette=color_palette)
plt.show()

cum_depth = np.zeros(n_clus)
num_points_cluster = np.zeros(n_clus)
prob_in_stn_cluster = []
for i, label in enumerate(mmodel.labels_):
    depth = df['depth'].iloc[i]

    if depth <= 1.75 and depth >= -4:
        prob_in_stn_cluster.append(label)

    cum_depth[label] += depth
    num_points_cluster[label] += 1

avg_depth = cum_depth / num_points_cluster
print("Cluster Sizes: " + str(num_points_cluster))
print("Average depth per cluster: " + str(avg_depth))
print("Cluster ids of points with -4 <= x <= 1.75: " + str(prob_in_stn_cluster))

sample_idxs = []
for idx, d in enumerate(df_lfp['aligned depth']):
    if d <= 1.75 and d >= -4 :
        sample_idxs.append(idx)

for i in range(0, 668):
    n = random.randint(0, df.shape[0] - 1)
    while n in sample_idxs:
        n = random.randint(0, df.shape[0] - 1)

    sample_idxs.append(n)

sys.exit()

feature_analyzer = FeatureAnalyzer(df, sample_idxs, mmodel, mmodel.fit_predict,
                                   list(df)[:-2], list(df)[-1], ModelType.CLUSTER)

print("=== SHAP Plots ===")
feature_analyzer.bar_plot()
feature_analyzer.beeswarm_plot()
feature_analyzer.heatmap_plot()
feature_analyzer.decision_plot()

print("=== Feature Cov matrix ===")
feature_analyzer.correlation_matrix_plot()

print("=== Shap Selection ===")
feature_analyzer.shap_select()

# mmodel = xgboost.XGBRegressor(n_jobs=multiprocessing.cpu_count() // 2, max_depth=5)
# mmodel.fit(df.iloc[:, :-1], df.iloc[:, -1])
# xgboost.plot_importance(mmodel)
# xgboost.plot_tree(mmodel)
# 
# sample_idxs = []
# for idx, d in enumerate(df_lfp['aligned depth']):
#     if d <= 1.75 and d >= -4 :
#         sample_idxs.append(idx)
# 
# for i in range(0, 333):
#     n = random.randint(0, df.shape[0] - 1)
#     while n in sample_idxs:
#         n = random.randint(0, df.shape[0] - 1)
# 
#     sample_idxs.append(n)
# 
# feature_analyzer = FeatureAnalyzer(df, sample_idxs, mmodel, mmodel.predict,
#                                    list(df)[:-1], list(df)[-1], ModelType.TREE)
# 
# print("=== SHAP Plots ===")
# feature_analyzer.bar_plot()
# feature_analyzer.beeswarm_plot()
# feature_analyzer.heatmap_plot()
# feature_analyzer.decision_plot()
# 
# print("=== Feature Cov matrix ===")
# feature_analyzer.correlation_matrix_plot()
# 
# print("=== Shap Selection ===")
# feature_analyzer.shap_select()

############### CLASSIFICATION based on eyeballed limits [-4, 1.75] ################################
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
#print("=== Shap Selection ===")
#feature_analyzer.shap_select()


########################### REGRESSION of aligned depth ######################################
# df['depth'] = df_lfp['aligned depth']
# 
# mmodel = xgboost.XGBRegressor(n_jobs=multiprocessing.cpu_count() // 2, max_depth=5)
# mmodel.fit(df.iloc[:, :-1], df.iloc[:, -1])
# xgboost.plot_importance(mmodel)
# xgboost.plot_tree(mmodel)
# 
# sample_idxs = []
# for idx, d in enumerate(df_lfp['aligned depth']):
#     if d <= 1.75 and d >= -4 :
#         sample_idxs.append(idx)
# 
# for i in range(0, 333):
#     n = random.randint(0, df.shape[0] - 1)
#     while n in sample_idxs:
#         n = random.randint(0, df.shape[0] - 1)
# 
#     sample_idxs.append(n)
# 
# feature_analyzer = FeatureAnalyzer(df, sample_idxs, mmodel, mmodel.predict,
#                                    list(df)[:-1], list(df)[-1], ModelType.TREE)
# 
# print("=== SHAP Plots ===")
# feature_analyzer.bar_plot()
# feature_analyzer.beeswarm_plot()
# feature_analyzer.heatmap_plot()
# feature_analyzer.decision_plot()
# 
# print("=== Feature Cov matrix ===")
# feature_analyzer.correlation_matrix_plot()
# 
# print("=== Shap Selection ===")
# feature_analyzer.shap_select()
# 
