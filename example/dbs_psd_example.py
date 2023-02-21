import random
from collections import Counter
import math
import pickle
import sys
from io import TextIOWrapper
from typing import Final
import multiprocessing

from concept_formation import cluster
from concept_formation.trestle import TrestleTree
from concept_formation.visualize import visualize as viz_tree
from hdbscan import HDBSCAN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import (OPTICS, KMeans, DBSCAN, SpectralClustering,
                             MeanShift, AffinityPropagation, BisectingKMeans)
import sklearn
from sklearn.experimental import enable_halving_search_cv
from umap import UMAP
import xgboost

sys.path.append(r"/home/someusername/sync/workspace/nb_tue/3/2_gharabaghi/"
                r"data-driven-biomarker/src/")
sys.path.append(r"/home/someusername/sync/workspace/nb_tue/3/2_gharabaghi/"
                r"intraop-neuro-phys-dataset_private")
from ddbm import FeatureAnalyzer, ModelType


N_NON_FEAT_COLS: Final[int] = 6


def preprocess(path: str, group_sz: int, flattened: bool, fname: str) -> None:
    with open(path, "rb") as group_file:
        full_df = pickle.load(group_file)

    dbs_df = full_df.dbs_features

    flat_str = '_flat' if flattened else ''
    freqs = dbs_df[f'dbs{flat_str}_psds_freqs'].iloc[0]
    slim_df = dbs_df[['anatomical_struct', 'lead_dbs_coords', 'aligned_depth',
             f'dbs{flat_str}_psds', 'op_date', 'dbs_exponents', 'dbs_mads']]

    # flatten the psd from a list in one column to one column per frequency
    n_df = slim_df[f'dbs{flat_str}_psds'].apply(pd.Series)
    n_df.columns = map(str, map(int, freqs))

    binned_df = pd.DataFrame()
    col_segments = range(math.ceil(n_df.shape[1] / float(group_sz)))
    for col in col_segments:
        fst_idx = f"{col * group_sz + freqs[0]}_"
        snd_idx = f"{(col + 1) * group_sz + freqs[0] - 1}_Hz" if col != col_segments[-1] else f"{freqs[-1]}_Hz"
        idx = (f"{fst_idx}{snd_idx}")
        binned_df[idx] = (n_df.iloc[:, group_sz * col : group_sz * (col + 1)]
                           .mean(axis=1))

    binned_df['op_date'] = slim_df['op_date']
    binned_df['depth'] = slim_df['aligned_depth']
    binned_df['coords'] = slim_df['lead_dbs_coords']
    binned_df['structure'] = slim_df['anatomical_struct']
    binned_df['in_stn'] = [1 if any('STN_motor' in s for s in structs) else 0
                           for structs in binned_df['structure']]
    binned_df['cluster'] = np.zeros(slim_df.shape[0])
    binned_df['stn_not_motor'] = [1 if any('STN' in s or 'SN' in s for s in structs) and not in_target 
                                  else 0 for structs, in_target in zip(binned_df['structure'], binned_df['in_stn'])]
    #[1 if any(('SNr' in s or 'SNc' in s or 'STN' in s) and 'STN_motor' not in s for s in structs)
                  #   else 0 for structs in binned_df['structure']]
    binned_df = binned_df[binned_df['stn_not_motor'] == 0]
    binned_df.drop(columns=['stn_not_motor'], inplace=True)
    
    del n_df
    del slim_df
    slim_df = binned_df
    slim_df.reset_index(inplace=True, drop=True)
    slim_df.to_hdf(fname, 'feats')


def visualize(df: pd.DataFrame,
              method: str = 'pca',
              n_patients: int = 25
              ) -> np.ndarray:
    assert method in ['pca', 'umap', 'tsne'], ("method must be one of 'pca',"
                                               "'umap', 'tsne'")


    print(df.iloc[:, 0: -N_NON_FEAT_COLS])

    # Transform data for 3d plotting
    if method == 'pca':
        data = sklearn.decomposition.PCA(n_components=3).fit_transform(
                df.iloc[:, 0: -N_NON_FEAT_COLS])
    elif method == 'tsne':
        data = sklearn.manifold.TSNE(n_components=3).fit_transform(
                df.iloc[:, 0: -N_NON_FEAT_COLS])
    elif method == 'umap':
        data = UMAP(n_components=3).fit_transform(
                df.iloc[:, 0: -N_NON_FEAT_COLS])

    # Color points red if they are in some STN subregion, grey otherwise
    in_stn_colors = [ (1, 0, 0) if d == 1 else (0.5, 0.5, 0.5)
                     for d in df['in_stn']]
    # plot all patients
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(xs=data.T[0], ys=data.T[1], zs=data.T[2], linewidth=0,
               c=in_stn_colors, alpha=0.5)
    plt.show()
    plt.close('all')

    # plot the rows corresponding to the first n_patients, colors as before
    i = 1
    prev_op_date = df['op_date'].iloc[0]
    patient_colors = []
    fig = plt.figure()
    pat_root = int(np.ceil(np.sqrt(n_patients)))
    for row in range(len(data.T[0])):
        if prev_op_date != df['op_date'].iloc[row]:
            ax = fig.add_subplot(pat_root, pat_root, i, projection='3d')
            ax.scatter(xs=data.T[0][row - len(patient_colors) : row],
                       ys=data.T[1][row - len(patient_colors) : row],
                       zs=data.T[2][row - len(patient_colors) : row],
                       linewidth=0, c=patient_colors, alpha=0.7)
            i += 1
            patient_colors = []
            if i > n_patients:
                break

        if df['in_stn'].iloc[row] == 1:
            patient_colors.append((1, 0, 0))
        else:
            patient_colors.append((0.5, 0.5, 0.5))

        prev_op_date = df['op_date'].iloc[row]

    plt.show()
    plt.close('all')

    return data

def describe(df: pd.DataFrame, wfile: TextIOWrapper):
    wfile.write(f'{df.columns}\n')
    wfile.write(f'{df.describe()}\n')
    wfile.write(f'# Patients: {df["op_date"].nunique()}, #samples: {df.shape[0]}, depth range: {df["depth"].min()} - {df["depth"].max()} mm, #inside motor STN {df["in_stn"].value_counts()}\n')
    wfile.write(f'{df["structure"].value_counts()}\n')
    wfile.write(f'{df["depth"].value_counts()}\n')


def clustering(df: pd.DataFrame,
               sample_idxs: list[int],
               wfile: TextIOWrapper,
               stats: bool = True,
               plot: bool = False,
               proj_data: np.ndarray = None):
    # -6 for cluster, in stn, anatomical structure, coords, depth, op_date
    for algo in [KMeans, BisectingKMeans, SpectralClustering, HDBSCAN, OPTICS,
                 DBSCAN, MeanShift, AffinityPropagation, TrestleTree]:
    # mmodel = hdbscan.HDBSCAN(min_samples=1, min_cluster_size=3,
    # cluster_selection_epsilon=1.4, cluster_selection_method='leaf')
        algo_str = str(str(algo).split('.')[-1]).split("'")[0]
        wfile.write(f'============ Clustering with {algo_str} ==============\n')
        if algo in [HDBSCAN, DBSCAN, OPTICS, MeanShift, AffinityPropagation,
                    TrestleTree]:
            mmodel = algo()
        else:
            params = {'n_clusters': range(2, 32)}
            mmodel = sklearn.model_selection.HalvingGridSearchCV(
                    algo(),
                    params,
                    scoring='adjusted_mutual_info_score')
            mmodel = mmodel.fit(df.iloc[:, :-N_NON_FEAT_COLS], df.iloc[:, -2])
            mmodel = mmodel.best_estimator_
            mmodel = algo(n_clusters=8)

        if algo is not TrestleTree:
            mmodel.fit(df.iloc[:, :-N_NON_FEAT_COLS])
            df['cluster'] = mmodel.labels_
            n_clus = mmodel.labels_.max() + 1
            df['cluster'] = df['cluster'].mask(df['cluster'] == -1, n_clus)
        else:
            mmodel.fit(df.iloc[:, :-N_NON_FEAT_COLS].to_dict('records'))
            clust = cluster.cluster(mmodel, df.iloc[:, :-N_NON_FEAT_COLS]
                                    .to_dict('records'))
            labels = list(clust)[0]
            le = sklearn.preprocessing.LabelEncoder().fit(labels)
            n_clus = len(le.classes_)
            df['cluster'] = le.transform(labels)
            if plot:
                viz_tree(mmodel, dst='trestle')

        if plot:
            assert proj_data is not None, ('proj_data must be provided for'
                                           ' plotting')
            color_palette = sns.color_palette(palette='bright',
                                              n_colors=n_clus + 1)
            cluster_colors = [color_palette[x] if x >= 0
                              else (0.5, 0.5, 0.5)
                              for x in df['cluster']]
            ax = plt.figure().add_subplot(projection='3d')
            ax.scatter(xs=proj_data.T[0], ys=proj_data.T[1], zs=proj_data.T[2],
                       linewidth=0, c=cluster_colors, alpha=0.5)
            plt.show()

        if stats:
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
            clst['%in_stn'] = [counts[i] / num_points_cluster[i]
                               for i in range(0, n_clus + 1)]
            in_stn_tot = df['in_stn'].value_counts()[1]
            clst['%stn_rows_in_clust'] = [counts[i]/in_stn_tot
                                          for i in range(0, n_clus + 1)]
            clst.to_csv(algo_str + '_clustering_comparison.csv')

        feature_analyzer = FeatureAnalyzer(df,
                                           sample_idxs,
                                           mmodel,
                                           None,
                                           list(df)[:-N_NON_FEAT_COLS],
                                           list(df)[-1],
                                           ModelType.CLUSTER)

        feature_analyzer.bar_plot(True, algo_str + '_shap_bar.png')
        feature_analyzer.beeswarm_plot(True, algo_str + '_shap_beeswarm.png')
        feature_analyzer.heatmap_plot(True, algo_str + '_shap_heatmap.png')
        feature_analyzer.decision_plot(True, algo_str + '_shap_decision.png')
        feature_analyzer.shap_select(log=True, wfile=wfile, sep=algo_str)
        feature_analyzer.correlation_matrix_plot(True,
                    f'{algo_str}_correlation_mat.png')


def classification(df: pd.DataFrame,
                   train: pd.DataFrame,
                   test: pd.DataFrame,
                   wfile: TextIOWrapper,
                   plot: bool = False,
                   proj_data: np.ndarray = None
                   ) -> None:
    wfile.write('Classification w DecisionTree\n')
    mmodel = sklearn.tree.DecisionTreeClassifier(max_depth=5)
    mmodel.fit(train.iloc[:, :-N_NON_FEAT_COLS], train.iloc[:, -2])

    wfile.write(f'Test Accuracy: {mmodel.score(test.iloc[:, :-N_NON_FEAT_COLS], test.iloc[:, -2])}\n')
    wfile.write(f'Whole Dataset Accuracy: {mmodel.score(df.iloc[:, :-N_NON_FEAT_COLS], df.iloc[:, -2])}\n')
    if plot:
        sklearn.tree.plot_tree(mmodel)
        plt.show()
        print(mmodel.feature_importances_)


        color_palette = sns.color_palette(palette='bright', n_colors=3)
        cluster_colors = [color_palette[int(x)]
                          for x in mmodel.predict(df.iloc[:, :-N_NON_FEAT_COLS])]
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(xs=proj_data.T[0], ys=proj_data.T[1], zs=proj_data.T[2],
                   linewidth=0, c=cluster_colors, alpha=1)
        plt.show()

    feature_analyzer = FeatureAnalyzer(df, test.index.values, mmodel, mmodel.predict,
                                       list(df)[:-N_NON_FEAT_COLS], list(df)[-2], ModelType.TREE)

    feature_analyzer.bar_plot(True, 'DecisionTree_shap_bar.png')
    feature_analyzer.beeswarm_plot(True, 'DecisionTree_shap_beeswarm.png')
    feature_analyzer.heatmap_plot(True, 'DecisionTree_shap_heatmap.png')
    feature_analyzer.decision_plot(True, 'DecisionTree_shap_decision.png')

    feature_analyzer.correlation_matrix_plot(True, 'DecisionTree_correlation_mat.png')

    feature_analyzer.shap_select(log=True, wfile=wfile, sep='DecisionTree')


def regression(df: pd.DataFrame,
               train: pd.DataFrame,
               test: pd.DataFrame,
               wfile: TextIOWrapper,
               plot: bool = False,
               proj_data: np.ndarray = None
               ) -> None:
    wfile.write('Regression w XGB\n')
    mmodel = xgboost.XGBRegressor(n_jobs=multiprocessing.cpu_count() // 2,
                                  max_depth=5)
    mmodel.fit(train.iloc[:, :-N_NON_FEAT_COLS],
               train.iloc[:, -5])

    test_r2 = sklearn.metrics.r2_score(
            test.iloc[:, -5],
            mmodel.predict(test.iloc[:, :-N_NON_FEAT_COLS]))

    whole_r2 = sklearn.metrics.r2_score(
           df.iloc[:, -5],
           mmodel.predict(df.iloc[:, :-N_NON_FEAT_COLS]))
    wfile.write(f'Test R^2: {test_r2}\n')
    wfile.write(f'Whole dataset R^2: {whole_r2}\n')

    xgboost.plot_importance(mmodel)
    xgboost.plot_tree(mmodel)
    plt.show()
    plt.close()

    feature_analyzer = FeatureAnalyzer(df, test.index.values, mmodel, mmodel.predict,
                                       list(df)[:-N_NON_FEAT_COLS],
                                       list(df)[-5], ModelType.TREE)

    feature_analyzer.bar_plot(True, 'XGBRegressor_shap_bar.png')
    feature_analyzer.beeswarm_plot(True, 'XGBRegressor_shap_beeswarm.png')
    feature_analyzer.heatmap_plot(True, 'XGBRegressor_shap_heatmap.png')
    feature_analyzer.decision_plot(True, 'XGBRegressor_shap_decision.png')

    feature_analyzer.correlation_matrix_plot(True, 'XGBRegressor_correlation_mat.png')

    feature_analyzer.shap_select(log=True, wfile=wfile, sep='XGBRegressor')



if __name__ == "__main__":
    pkl_path = "/run/media/someusername/RAID5Array/dbs-data/out/Group_at_1676482253.pkl"
    m_flat = True 
    m_flat_str = '_flat' if m_flat else ''
    m_hertz_bins = 5
    m_fname = f'dbs{m_flat_str}_psd_bin_{m_hertz_bins}_mad_exp.h5'

    preprocess(pkl_path, m_hertz_bins, m_flat, m_fname)
    m_df = pd.read_hdf(m_fname)
    # transformed_data = visualize(m_df, method='umap')

    m_wfile = open("stn_motor.log", 'w')
    describe(m_df, m_wfile)

    m_train, m_test = sklearn.model_selection.train_test_split(m_df,
                                                           test_size=0.2)
    classification(m_df, m_train, m_test, wfile=m_wfile, plot=False)

    m_wfile.close()

