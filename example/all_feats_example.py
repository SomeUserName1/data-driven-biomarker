import random
import math
from pprint import pprint
from collections import Counter
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
from sklearn.cluster import OPTICS, KMeans, DBSCAN, SpectralClustering, MeanShift, AffinityPropagation, BisectingKMeans
from hdbscan import HDBSCAN
#from Group import IntraOpNeuroDataset

pkl_path = "/home/someusername/sync/workspace/nb_tue/3/2_gharabaghi/group_1669237550.pkl"

with open(pkl_path, "rb") as group_file:
    intraop_data = pickle.load(group_file)

df = intraop_data.lfp_features_df

# Use green contacts only
freqs = df['lfp flat psd freqs'].iloc[0]
df = df.loc[df['contact'] == 0][['aligned depth', 'lfp flat psd']]

# flatten the psd from a list in one column to one column per frequency
n_df = df['lfp flat psd'].apply(pd.Series)
n_df.columns = map(str, map(int, freqs))

group_sz = 3
binned_df = pd.DataFrame()
col_segments = range(math.ceil(n_df.shape[1] / float(group_sz)))
for col in col_segments:
    binned_df[str(col * group_sz + freqs[0]) + "_" + str((col + 1) * group_sz + freqs[0] - 1) + '_Hz'] = n_df.iloc[:, group_sz * col : group_sz * (col + 1)].mean(axis=1)

binned_df['depth'] = df['aligned depth']
binned_df['in_stn'] = [1 if d <= 1.75 and d >= -4 else 0 for d in binned_df['depth']]
binned_df['cluster'] = np.zeros(df.shape[0])
del n_df
del df
del intraop_data
df = binned_df
df.reset_index(inplace=True, drop=True)

sample_idxs = df.index[df['in_stn'] == 1].to_list()
len_trues = len(sample_idxs)
for i in range(0, len_trues):
    n = random.randint(0, df.shape[0] - 1)
    while n in sample_idxs:
        n = random.randint(0, df.shape[0] - 1)

    sample_idxs.append(n)

#print(df.describe())
#print(df.head(n=10))
#print(df.groupby(['in_stn']).count())
data = UMAP(n_components=3).fit_transform(df.iloc[:, :-3])
in_stn_colors = [ (1, 0, 0) if d == 1 else (0.5, 0.5, 0.5) for d in df['in_stn']]

ax = plt.figure().add_subplot(projection='3d')
ax.scatter(xs=data.T[0], ys=data.T[1], zs=data.T[2], linewidth=0, c=in_stn_colors, alpha=0.5)
plt.show()
plt.clf()

############### CLUSTERING of PSDs per depth #######################################################
#for algo in [HDBSCAN, OPTICS, DBSCAN, MeanShift, AffinityPropagation, KMeans, BisectingKMeans, SpectralClustering]:
## mmodel = hdbscan.HDBSCAN(min_samples=1, min_cluster_size=3, cluster_selection_epsilon=1.4, cluster_selection_method='leaf')
#    algo_str = str(str(algo).split('.')[-1]).split("'")[0]
#    print('============ ' + algo_str + ' ==============')
#    if algo in [HDBSCAN, DBSCAN, OPTICS, MeanShift, AffinityPropagation]:
#        mmodel = algo()
#    else:
#        mmodel = algo(n_clusters=2)
#
#    mmodel.fit(df.iloc[:, :-3])
#
#    df['cluster'] = mmodel.labels_
#    n_clus = mmodel.labels_.max() + 1
#    df['cluster'] = df['cluster'].mask(df['cluster'] == -1, n_clus)
#
#    color_palette = sns.color_palette('Paired', n_clus)
#    cluster_colors = [color_palette[x] if x >= 0
#                      else (0.5, 0.5, 0.5)
#                      for x in mmodel.labels_]
#    ax = plt.figure().add_subplot(projection='3d')
#    ax.scatter(xs=data.T[0], ys=data.T[1], zs=data.T[2], linewidth=0, c=cluster_colors, alpha=0.5) 
#    plt.savefig('clust_scatter_' + algo_str + '.png')
#    plt.clf()
#
#    cum_depth = np.zeros(n_clus + 1)
#    num_points_cluster = np.zeros(n_clus + 1)
#    prob_in_stn_cluster = []
#    for i in range(df.shape[0]):
#        label = df['cluster'].iloc[i]
#
#        if df['in_stn'].iloc[i]:
#            prob_in_stn_cluster.append(label)
#
#        cum_depth[label] += df['depth'].iloc[i]
#        num_points_cluster[label] += 1
#
#    avg_depth = cum_depth / num_points_cluster
#    counts = Counter(prob_in_stn_cluster)
#
#    clst = pd.DataFrame()
#    clst['size'] = num_points_cluster
#    clst['avg_depth'] = avg_depth
#    clst['#in_stn'] = [counts[i] for i in range(0, n_clus + 1)]
#    clst['%in_stn'] = [counts[i] / num_points_cluster[i] for i in range(0, n_clus + 1)]
#    in_stn_tot = df['in_stn'].value_counts()[1]
#    clst['%stn_rows_in_clust'] = [counts[i]/in_stn_tot for i in range(0, n_clus + 1)]
#    clst.to_csv('clustering_comparison_' + algo_str + '.csv')
#
#    feature_analyzer = FeatureAnalyzer(df, sample_idxs, mmodel, mmodel.fit_predict,
#                                       list(df)[:-3], list(df)[-1], ModelType.CLUSTER)
#
#    feature_analyzer.bar_plot()
#    feature_analyzer.beeswarm_plot()
#    feature_analyzer.heatmap_plot()
#    feature_analyzer.decision_plot()
#
#    print("=== Shap Selection ===")
#    feature_analyzer.shap_select()
#
#feature_analyzer.correlation_matrix_plot()


############### CLASSIFICATION ###############################
mmodel = sklearn.tree.DecisionTreeClassifier(max_depth=5)
mmodel.fit(df.iloc[:, :-3], df.iloc[:, -2])
sklearn.tree.plot_tree(mmodel)
plt.show()
print(mmodel.feature_importances_)

feature_analyzer = FeatureAnalyzer(df, sample_idxs, mmodel, mmodel.predict,
                                   list(df)[:-3], list(df)[-2], ModelType.TREE)

print("=== SHAP Plots ===")
feature_analyzer.bar_plot()
feature_analyzer.beeswarm_plot()
feature_analyzer.heatmap_plot()
feature_analyzer.decision_plot()

print("=== Feature Cov matrix ===")
feature_analyzer.correlation_matrix_plot()

print("=== Shap Selection ===")
feature_analyzer.shap_select()

########################### REGRESSION ######################################
mmodel = xgboost.XGBRegressor(n_jobs=multiprocessing.cpu_count() // 2, max_depth=5)
mmodel.fit(df.iloc[:, :-3], df.iloc[:, -3])
xgboost.plot_importance(mmodel)
xgboost.plot_tree(mmodel)
 
feature_analyzer = FeatureAnalyzer(df, sample_idxs, mmodel, mmodel.predict,
                                   list(df)[:-3], list(df)[-3], ModelType.TREE)

print("=== SHAP Plots ===")
feature_analyzer.bar_plot()
feature_analyzer.beeswarm_plot()
feature_analyzer.heatmap_plot()
feature_analyzer.decision_plot()

print("=== Feature Cov matrix ===")
feature_analyzer.correlation_matrix_plot()

print("=== Shap Selection ===")
feature_analyzer.shap_select()


#### recluster only in stn and analyze features to detect sub regions of the stn like motor & cognitive
