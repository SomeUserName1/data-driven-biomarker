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
# from umap import UMAP
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
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from hdbscan import HDBSCAN
from sklearn.model_selection import train_test_split
from concept_formation.trestle import TrestleTree
from concept_formation import cluster
from concept_formation.visualize import visualize as viz_tree
from concept_formation.visualize import visualize_clusters
import tabloo
#from Group import IntraOpNeuroDataset

################ Extract used features from full df ################
pkl_path = "/run/media/someusername/RAID5Array/dbs-data/out/Group_at_1675947461.pkl"

with open(pkl_path, "rb") as group_file:
    df = pickle.load(group_file)

df = df.dbs_features
print(df.describe())
print(df.head(n=10))
print(df.columns)
print(df['dbs_exponents'])
for col in df.columns:
    print(f"{col}, {df[col].iloc[0]}")

# Use green contacts only
freqs = df['dbs_flat_psds_freqs'].iloc[0]
df = df[['anatomical_struct', 'lead_dbs_coords', 'aligned_depth',
         'dbs_flat_psds', 'op_date', 'dbs_exponents', 'dbs_mads']]

# flatten the psd from a list in one column to one column per frequency
n_df = df['dbs_flat_psds'].apply(pd.Series)
n_df.columns = map(str, map(int, freqs))

group_sz = 5
binned_df = pd.DataFrame()
col_segments = range(math.ceil(n_df.shape[1] / float(group_sz)))
for col in col_segments:
    idx = (f"{col * group_sz * freqs[0]}_"
               "{(col + 1) * group_sz * freqs[0] - 1}_Hz")
    binned_df[idx] = (n_df.iloc[:, group_sz * col : group_sz * (col + 1)]
                       .mean(axis=1))

binned_df['mad'] = df['dbs_mads']
binned_df['exponent'] = df['dbs_exponents']
binned_df['op_date'] = df['op_date']
binned_df['depth'] = df['aligned_depth']
binned_df['coords'] = df['lead_dbs_coords']
binned_df['structure'] = df['anatomical_struct']

del n_df
del df
df = binned_df
df.reset_index(inplace=True, drop=True)
df.to_hdf('dbs_bin5_new_df.h5', 'feats')

########### add stn label, sample, Visualize #############
df = pd.read_hdf('dbs_bin5_new_df.h5')

df['in_stn'] = [1 if 'STN' in s else 0 for s in df['structure']]
sample_idxs = df.index[df['in_stn'] == 1].to_list()
len_trues = len(sample_idxs)
for i in range(0, len_trues):
    n = random.randint(0, df.shape[0] - 1)
    while n in sample_idxs:
        n = random.randint(0, df.shape[0] - 1)

    sample_idxs.append(n)

#print(df.describe())
#print(df.head(n=10))
print(len_trues)
data = TSNE(n_components=3).fit_transform(df.iloc[:, 0:-5])
# data = sklearn.decomposition.PCA(n_components=3).fit_transform(df.iloc[:, 1:-3])
in_stn_colors = [ (1, 0, 0) if d == 1 else (0.5, 0.5, 0.5) for d in df['in_stn']]

ax = plt.figure().add_subplot(projection='3d')
ax.scatter(xs=data.T[0], ys=data.T[1], zs=data.T[2], linewidth=0,
           c=in_stn_colors, alpha=0.5)
plt.show()
plt.close('all')

# plot per patient
i = 1
prev_op_date = df['OP date'].iloc[0]
n_pats = len(df['OP date'].unique())
patient_colors = []
fig = plt.figure()
for row in range(len(data.T[0])):
    if prev_op_date != df['OP date'].iloc[row]:
        ax = fig.add_subplot(5, 5, i, projection='3d')
        ax.scatter(xs=data.T[0][row - len(patient_colors) : row],
                   ys=data.T[1][row - len(patient_colors) : row],
                   zs=data.T[2][row - len(patient_colors) : row],
                   linewidth=0, c=patient_colors, alpha=1)
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

df = df.drop(['OP date'], axis=1)
df['cluster'] = np.zeros(df.shape[0])
wfile = open("shap_based_feature_selection.txt", 'w')

############### CLUSTERING of PSDs per depth #######################################################
# -5 for cluster, in stn, anatomical structure, coords, depth
for algo in [KMeans, BisectingKMeans, HDBSCAN, OPTICS, DBSCAN, MeanShift, AffinityPropagation, TrestleTree]:
# mmodel = hdbscan.HDBSCAN(min_samples=1, min_cluster_size=3, cluster_selection_epsilon=1.4, cluster_selection_method='leaf')
    algo_str = str(str(algo).split('.')[-1]).split("'")[0]
    print('============ ' + algo_str + ' ==============')
    if algo in [HDBSCAN, DBSCAN, OPTICS, MeanShift, AffinityPropagation, TrestleTree]:
        mmodel = algo()
    else:
        params = {'n_clusters': range(2, 32)}
        mmodel = HalvingGridSearchCV(algo(), params, scoring='adjusted_mutual_info_score').fit(df.iloc[:, :-5], df.iloc[:, -2])
        mmodel = mmodel.best_estimator_
        mmodel = algo(n_clusters=8)

    if algo is not TrestleTree:
        mmodel.fit(df.iloc[:, :-5])
        df['cluster'] = mmodel.labels_
        n_clus = mmodel.labels_.max() + 1
        df['cluster'] = df['cluster'].mask(df['cluster'] == -1, n_clus)
    else:
        mmodel.fit(df.iloc[:, :-5].to_dict('records'))
        clust = cluster.cluster(mmodel, df.iloc[:, :-5].to_dict('records'))
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

    cum_x = np.zeros(n_clus + 1)
    cum_y = np.zeros(n_clus + 1)
    cum_z = np.zeros(n_clus + 1)
    num_points_cluster = np.zeros(n_clus + 1)
    prob_in_stn_cluster = []
    for i in range(df.shape[0]):
        label = df['cluster'].iloc[i]

        if df['in_stn'].iloc[i]:
            prob_in_stn_cluster.append(label)

        cum_x[label] += df['coords'].iloc[i][0]
        cum_y[label] += df['coords'].iloc[i][1]
        cum_z[label] += df['coords'].iloc[i][2]
        num_points_cluster[label] += 1

    avg_x = cum_x / num_points_cluster
    avg_y = cum_y / num_points_cluster
    avg_z = cum_z / num_points_cluster
    counts = Counter(prob_in_stn_cluster)

    clst = pd.DataFrame()
    clst['size'] = num_points_cluster
    clst['avg_x'] = avg_x
    clst['avg_y'] = avg_y
    clst['avg_z'] = avg_z
    clst['#in_stn'] = [counts[i] for i in range(0, n_clus + 1)]
    clst['%in_stn'] = [counts[i] / num_points_cluster[i] for i in range(0, n_clus + 1)]
    in_stn_tot = df['in_stn'].value_counts()[1]
    clst['%stn_rows_in_clust'] = [counts[i]/in_stn_tot for i in range(0, n_clus + 1)]
    clst.to_csv(algo_str + '_clustering_comparison.csv')

    feature_analyzer = FeatureAnalyzer(df, sample_idxs, mmodel, None,
                                       list(df)[:-5], list(df)[-1], ModelType.CLUSTER)

    feature_analyzer.bar_plot(True, algo_str + '_shap_bar.png')
    feature_analyzer.beeswarm_plot(True, algo_str + '_shap_beeswarm.png')
    feature_analyzer.heatmap_plot(True, algo_str + '_shap_heatmap.png')
    feature_analyzer.decision_plot(True, algo_str + '_shap_decision.png')

    feature_analyzer.shap_select(log=True, wfile=wfile, sep=algo_str)

    feature_analyzer.correlation_matrix_plot(True, algo_str + '_correlation_mat.png')


train, test = train_test_split(df, test_size=0.2, random_state=0)
print(train.columns)
print("############### CLASSIFICATION ###############################")
mmodel = sklearn.tree.DecisionTreeClassifier(max_depth=5)
mmodel.fit(train.iloc[:, :-5], train.iloc[:, -2])
print(mmodel.score(test.iloc[:, :-5], test.iloc[:, -2]))
print(mmodel.score(df.iloc[:, :-5], df.iloc[:, -2]))
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
cluster_colors = [color_palette[int(x)]
                  for x in mmodel.predict(df.iloc[:, :-5])]
ax = plt.figure().add_subplot(projection='3d')
ax.scatter(xs=data.T[0], ys=data.T[1], zs=data.T[2], linewidth=0, c=cluster_colors, alpha=1)
plt.show()


print("########################### REGRESSION ######################################")
mmodel = xgboost.XGBRegressor(n_jobs=multiprocessing.cpu_count() // 2, max_depth=5)
mmodel.fit(train.iloc[:, :-3], train.iloc[:, -5])
print(sklearn.metrics.r2_score(test.iloc[:, -5], mmodel.predict(test.iloc[:, :-5])))
print(sklearn.metrics.r2_score(df.iloc[:, -5], mmodel.predict(df.iloc[:, :-5])))
xgboost.plot_importance(mmodel)
xgboost.plot_tree(mmodel)

feature_analyzer = FeatureAnalyzer(df, sample_idxs, mmodel, mmodel.predict,
                                   list(df)[:-5], list(df)[-5], ModelType.TREE)

feature_analyzer.bar_plot(True, 'XGBRegressor_shap_bar.png')
feature_analyzer.beeswarm_plot(True, 'XGBRegressor_shap_beeswarm.png')
feature_analyzer.heatmap_plot(True, 'XGBRegressor_shap_heatmap.png')
feature_analyzer.decision_plot(True, 'XGBRegressor_shap_decision.png')

feature_analyzer.correlation_matrix_plot(True, 'XGBRegressor_correlation_mat.png')

feature_analyzer.shap_select(log=True, wfile=wfile, sep='XGBRegressor')

wfile.close()

#### recluster only in stn and analyze features to detect sub regions of the stn like motor & cognitive
