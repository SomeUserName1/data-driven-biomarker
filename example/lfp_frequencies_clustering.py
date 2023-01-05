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
from sklearn.preprocessing import LabelEncoder
import xgboost
from ddbm import FeatureAnalyzer, ModelType
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.cluster import OPTICS, KMeans, DBSCAN, SpectralClustering, MeanShift, AffinityPropagation, BisectingKMeans
from hdbscan import HDBSCAN
from concept_formation.trestle import TrestleTree
from concept_formation import cluster
from concept_formation.visualize import visualize as viz_tree
from concept_formation.visualize import visualize_clusters
#from Group import IntraOpNeuroDataset

pkl_path = "/home/someusername/sync/workspace/nb_tue/3/2_gharabaghi/group_1669237550.pkl"

with open(pkl_path, "rb") as group_file:
    intraop_data = pickle.load(group_file)

df = intraop_data.lfp_features_df

# Use green contacts only
freqs = df['lfp flat psd freqs'].iloc[0]
df = df.loc[df['contact'] == 0][['aligned depth', 'lfp flat psd', 'OP date']]



# flatten the psd from a list in one column to one column per frequency
n_df = df['lfp flat psd'].apply(pd.Series)
n_df.columns = map(str, map(int, freqs))

group_sz = 3
binned_df = pd.DataFrame()
col_segments = range(math.ceil(n_df.shape[1] / float(group_sz)))
for col in col_segments:
     binned_df[str(col * group_sz + freqs[0]) + "_" + str((col + 1) * group_sz + freqs[0] - 1) + '_Hz'] = n_df.iloc[:, group_sz * col : group_sz * (col + 1)].mean(axis=1)

for col in range(binned_df.shape[1]):
    binned_df.iloc[:, col] = (binned_df.iloc[:, col] - binned_df.iloc[:, col].min()) / (binned_df.iloc[:, col].max() - binned_df.iloc[:, col].min()) 

binned_df = n_df
binned_df['depth'] = df['aligned depth']
binned_df['OP date'] = df['OP date']
binned_df['in_stn'] = [1 if d <= 1.75 and d >= -4 else 0 for d in binned_df['depth']]
binned_df['cluster'] = np.zeros(df.shape[0])
del n_df
del df
del intraop_data
df = binned_df[binned_df['depth'] > -4]
df = df[df['depth'] < 8]
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
# data = UMAP(n_components=3).fit_transform(df.iloc[:, :-3])
data = sklearn.decomposition.PCA(n_components=3).fit_transform(df.iloc[:, :-4])
in_stn_colors = [ (1, 0, 0) if d == 1 else (0.5, 0.5, 0.5) for d in df['in_stn']]

ax = plt.figure().add_subplot(projection='3d')
ax.scatter(xs=data.T[0], ys=data.T[1], zs=data.T[2], linewidth=0, c=in_stn_colors, alpha=0.5)
plt.show()
plt.close('all')

i = 1
prev_op_date = df['OP date'].iloc[0]
n_pats = len(df['OP date'].unique())
patient_colors = []
fig = plt.figure()
for row in range(len(data.T[0])):
    if prev_op_date != df['OP date'].iloc[row]:
        ax = fig.add_subplot(5, 5, i, projection='3d')
        ax.scatter(xs=data.T[0][row - len(patient_colors) : row], ys=data.T[1][row - len(patient_colors) : row], zs=data.T[2][row - len(patient_colors) : row], linewidth=0, c=patient_colors, alpha=1)
        i += 1
        patient_colors = []
        if i > 25:
            break
        
    if df['in_stn'].iloc[row] == 1:
        patient_colors.append((1,0,0))
    else:
        patient_colors.append((0,1,0))

    prev_op_date = df['OP date'].iloc[row]

plt.show()
plt.close('all')


# FIXME remove op date or adjust indexes
wfile = open("shap_based_feature_selection.txt", 'a')
############### CLUSTERING of PSDs per depth #######################################################
for algo in [KMeans, BisectingKMeans, HDBSCAN, OPTICS, DBSCAN, MeanShift, AffinityPropagation, TrestleTree]:
# mmodel = hdbscan.HDBSCAN(min_samples=1, min_cluster_size=3, cluster_selection_epsilon=1.4, cluster_selection_method='leaf')
    algo_str = str(str(algo).split('.')[-1]).split("'")[0]
    print('============ ' + algo_str + ' ==============')
    if algo in [HDBSCAN, DBSCAN, OPTICS, MeanShift, AffinityPropagation, TrestleTree]:
        mmodel = algo()
    else:
        params = {'n_clusters': range(2, 32)}
        # mmodel = HalvingGridSearchCV(algo(), params, scoring='adjusted_mutual_info_score').fit(df.iloc[:, :-3], df.iloc[:, -2])
        # mmodel = mmodel.best_estimator_
        mmodel = algo(n_clusters=8)

    if algo is not TrestleTree:
        mmodel.fit(df.iloc[:, :-3])
        df['cluster'] = mmodel.labels_
        n_clus = mmodel.labels_.max() + 1
        df['cluster'] = df['cluster'].mask(df['cluster'] == -1, n_clus)
    else:
        mmodel.fit(df.iloc[:, :-3].to_dict('records'))
        clust = cluster.cluster(mmodel, df.iloc[:, :-3].to_dict('records'))
        labels = list(clust)[0]
        le = LabelEncoder().fit(labels)
        n_clus = len(le.classes_)
        df['cluster'] = le.transform(labels)
        viz_tree(mmodel, dst='trestle')

    color_palette = sns.color_palette(palette='bright', n_colors=n_clus+1)
    cluster_colors = [color_palette[x] if x >= 0
                      else (0.5, 0.5, 0.5)
                      for x in df['cluster']]
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(xs=data.T[0], ys=data.T[1], zs=data.T[2], linewidth=0, c=cluster_colors, alpha=0.5)
    plt.show()

    cum_depth = np.zeros(n_clus + 1)
    num_points_cluster = np.zeros(n_clus + 1)
    prob_in_stn_cluster = []
    for i in range(df.shape[0]):
        label = df['cluster'].iloc[i]

        if df['in_stn'].iloc[i]:
            prob_in_stn_cluster.append(label)

        cum_depth[label] += df['depth'].iloc[i]
        num_points_cluster[label] += 1

    avg_depth = cum_depth / num_points_cluster
    counts = Counter(prob_in_stn_cluster)

    clst = pd.DataFrame()
    clst['size'] = num_points_cluster
    clst['avg_depth'] = avg_depth
    clst['#in_stn'] = [counts[i] for i in range(0, n_clus + 1)]
    clst['%in_stn'] = [counts[i] / num_points_cluster[i] for i in range(0, n_clus + 1)]
    in_stn_tot = df['in_stn'].value_counts()[1]
    clst['%stn_rows_in_clust'] = [counts[i]/in_stn_tot for i in range(0, n_clus + 1)]
    clst.to_csv(algo_str + '_clustering_comparison.csv')

    feature_analyzer = FeatureAnalyzer(df, sample_idxs, mmodel, None,
                                       list(df)[:-3], list(df)[-1], ModelType.CLUSTER)

    feature_analyzer.bar_plot(True, algo_str + '_shap_bar.png')
    feature_analyzer.beeswarm_plot(True, algo_str + '_shap_beeswarm.png')
    feature_analyzer.heatmap_plot(True, algo_str + '_shap_heatmap.png')
    feature_analyzer.decision_plot(True, algo_str + '_shap_decision.png')

    feature_analyzer.shap_select(log=True, wfile=wfile, sep=algo_str)

    feature_analyzer.correlation_matrix_plot(True, algo_str + '_correlation_mat.png')


print("############### CLASSIFICATION ###############################")
mmodel = sklearn.tree.DecisionTreeClassifier(max_depth=5)
mmodel.fit(df.iloc[:, :-3], df.iloc[:, -2])
print(mmodel.score(df.iloc[:, :-3], df.iloc[:, -2]))
sklearn.tree.plot_tree(mmodel)
plt.show()
print(mmodel.feature_importances_)

feature_analyzer = FeatureAnalyzer(df, sample_idxs, mmodel, mmodel.predict,
                                   list(df)[:-3], list(df)[-2], ModelType.TREE)

feature_analyzer.bar_plot(True, 'DecisionTree_shap_bar.png')
feature_analyzer.beeswarm_plot(True, 'DecisionTree_shap_beeswarm.png')
feature_analyzer.heatmap_plot(True, 'DecisionTree_shap_heatmap.png')
feature_analyzer.decision_plot(True, 'DecisionTree_shap_decision.png')

feature_analyzer.correlation_matrix_plot(True, 'DecisionTree_correlation_mat.png')

feature_analyzer.shap_select(log=True, wfile=wfile, sep='DecisionTree')

color_palette = sns.color_palette(palette='bright', n_colors=3)
cluster_colors = [color_palette[x] 
                  for x in mmodel.predict(df.iloc[:, :-3])]
ax = plt.figure().add_subplot(projection='3d')
ax.scatter(xs=data.T[0], ys=data.T[1], zs=data.T[2], linewidth=0, c=cluster_colors, alpha=1)
plt.show()


print("########################### REGRESSION ######################################")
mmodel = xgboost.XGBRegressor(n_jobs=multiprocessing.cpu_count() // 2, max_depth=5)
mmodel.fit(df.iloc[:, :-3], df.iloc[:, -3])
print(sklearn.metrics.r2_score(df.iloc[:, -3], mmodel.predict(df.iloc[:, :-3])))
xgboost.plot_importance(mmodel)
xgboost.plot_tree(mmodel)
 
feature_analyzer = FeatureAnalyzer(df, sample_idxs, mmodel, mmodel.predict,
                                   list(df)[:-3], list(df)[-3], ModelType.TREE)

feature_analyzer.bar_plot(True, 'XGBRegressor_shap_bar.png')
feature_analyzer.beeswarm_plot(True, 'XGBRegressor_shap_beeswarm.png')
feature_analyzer.heatmap_plot(True, 'XGBRegressor_shap_heatmap.png')
feature_analyzer.decision_plot(True, 'XGBRegressor_shap_decision.png')

feature_analyzer.correlation_matrix_plot(True, 'XGBRegressor_correlation_mat.png')

feature_analyzer.shap_select(log=True, wfile=wfile, sep='XGBRegressor')

wfile.close()

#### recluster only in stn and analyze features to detect sub regions of the stn like motor & cognitive
