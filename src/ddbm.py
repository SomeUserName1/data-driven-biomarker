from abc import ABC
from enum import Enum

import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost
import numpy as np
from sklearn.feature_selection import RFECV, VarianceThreshold, \
                                      SelectFromModel, \
                                      SequentialFeatureSelector
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegressionCV, LassoCV

class ModelType(Enum):
    SIMPLE = 0
    TREE = 1
    DEEP = 2
    CLUSTER = 3

class FeatureAnalyzer(ABC):
    def __init__(self, df, samples, model, predict_fn, predictors,
                 predicteds, model_type=ModelType.SIMPLE, max_display=None, *clust_params):
        self.model_type = model_type
        self.df = df
        self.samples = samples
        self.model = model
        self.predict_fn = predict_fn
        self.predictors = predictors
        self.predicteds = predicteds
        self.X = df[predictors].iloc[samples]
        self.max_display = int(np.ceil(0.5 * len(self.predictors))) if max_display is None else max_display

        if model_type == ModelType.CLUSTER:
            # fit a classifier
            # the cluster labels have to be predicteds here!
            clf = xgboost.XGBClassifier(use_label_encoder=False)
            clf.fit(self.df[predictors], self.df[predicteds])
            self.model = clf
            self.predict_fn = clf.predict
            self.clust = model

        # compute the SHAP values for the model
        self.explainer = shap.Explainer(self.predict_fn, self.X)
        self.shap_values = self.explainer(self.X)

    def remove_zero_variance_feats(self):
        selector = VarianceThreshold()
        selector.fit(self.df[self.predictors])
        print(selector.get_feature_names_out())

    def get_max_importance(self, model):
        return np.amax(self.shap_values.values, axis=0)

    def get_mean_importance(self, model):
        return np.mean(self.shap_values.values, axis=0)

    # Feature importance/coefficient- free methods
    def backwards_select(self):
        backward = SequentialFeatureSelector(self.model,
                                             n_features_to_select="auto",
                                             direction="backward", tol=0.01)
        backward.fit(self.df[self.predictors], self.df[self.predicteds])
        print(backward.get_feature_names_out())

    def forwards_select(self):
        forward = SequentialFeatureSelector(self.model,
                                            n_features_to_select="auto",
                                            direction="forward", tol=0.01)
        forward.fit(self.df[self.predictors], self.df[self.predicteds])
        print(forward.get_feature_names_out())

    # logit for classification, lasso for regression feature elimination
    def logit_rfe_select(self):
        logit = LogisticRegressionCV(penalty='l1', solver='saga', n_jobs=-1)
        rfecv = RFECV(estimator=logit, n_jobs=-1)
        rfecv.fit(self.df[self.predictors], self.df[self.predicteds])
        print(rfecv.get_feature_names_out())
        print(rfecv.ranking_)

    def lasso_rfe_select(self):
        lasso = LassoCV(n_jobs=-1)
        rfecv = RFECV(estimator=lasso, n_jobs=-1)
        rfecv.fit(self.df[self.predictors], self.df[self.predicteds])
        print(rfecv.get_feature_names_out())
        print(rfecv.ranking_)

    def shap_select(self, log=False, wfile=None, sep=None, shap_max=False):
        fn = self.get_max_importance if shap_max else self.get_mean_importance
        selector = SelectFromModel(self.model, prefit=True, threshold="median",
                                   max_features=int(0.9 * len(self.predictors)),
                                   importance_getter=fn)
        selector.fit(self.df[self.predictors])

        if not log:
            print(selector.get_feature_names_out())
            print(fn(None)[selector.get_support(indices=True)])
        else:
            wfile.write(f"====== {sep} =======")
            wfile.write(str(selector.get_feature_names_out()))
            wfile.write(str(fn(None)[selector.get_support(indices=True)]))

    def correlation_matrix_plot(self, save=False, fig_path=None):
        plt.figure(figsize=(20,20))
        sns.heatmap(self.df.corr(), annot=True, cmap="RdYlGn")

        if save:
            if fig_path is None:
                fig_path = "correlation_matrix.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()

        plt.close('all')

#    def shap_interaction_values(self):
#        if self.model_type in [ModelType.TREE, ModelType.CLUSTER]:
#            print("Interaction values:")
#            print(self.explainer.shap_interaction_values(self.X))

    def partial_dependence_plot(self, col, save=False, fig_path=None):
    # make a standard partial dependence plot
        if not save:
            shap.plots.partial_dependence(
                col, self.predict_fn, self.X,
                model_expected_value=True, feature_expected_value=True,
                ice=False
            )
        else:
            shap.plots.partial_dependence(
                col, self.predict_fn, self.X,
                model_expected_value=True, feature_expected_value=True,
                ice=False, show=False
            )
            if fig_path is None:
                fig_path = "shap_part_dependence.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')

        plt.close('all')


    def scatter_plot(self, col, interaction=False, save=False, fig_path=None):
    # Shap value scatter plot
        if not save:
            shap.plots.scatter(self.shap_values[:, col], color=self.shap_values)
        else:
            shap.plots.scatter(self.shap_values[:, col], color=self.shap_values, show=False)
            if fig_path is None:
                fig_path = "shap_scatter.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')

        plt.close('all')


    def bar_plot(self, save=False, fig_path=None, cluster: bool = False, clustering_cutoff=0.8):
        if cluster:
            # Correlated features shown by dendrogram
            clustering = shap.utils.hclust(self.df[self.predictors],
                                           self.df[self.predicteds])
        else:
            clustering = None
        # Mean shap value bar plot
        if not save:
            shap.plots.bar(self.shap_values, max_display=self.max_display, clustering=clustering,
                       clustering_cutoff=clustering_cutoff)
        else:
            shap.plots.bar(self.shap_values, max_display=self.max_display, clustering=clustering,
                       clustering_cutoff=clustering_cutoff, show=False)
            if fig_path is None:
                fig_path = "shap_bar.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')

        plt.close('all')


    def waterfall_plot(self, index, save=False, fig_path=None):
    # the waterfall_plot shows how we get from shap_values.base_values to
    # model.predict(X)[sample_ind]
        if not save:
            shap.plots.waterfall(self.shap_values[index])
        else:
            shap.plots.waterfall(self.shap_values[index], show=False)
            if fig_path is None:
                fig_path = "shap_waterfall.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')

        plt.close('all')

    def beeswarm_plot(self, save=False, fig_path=None):
    # Beeswarm plot summarizes entire distribution of SHAP values f.a. features
        if not save:
            shap.plots.beeswarm(self.shap_values, max_display=self.max_display)
        else:
            shap.plots.beeswarm(self.shap_values, max_display=self.max_display, show=False)
            if fig_path is None:
                fig_path = "shap_beeswarm.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')

        plt.close('all')

    def heatmap_plot(self, save=False, fig_path=None):
    # Heatmaps show value distribution vs. shap value per feature
        if not save:
            shap.plots.heatmap(self.shap_values, max_display=self.max_display)
        else:
            shap.plots.heatmap(self.shap_values, max_display=self.max_display, show=False)
            if fig_path is None:
                fig_path = "shap_heatmap.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')

        plt.close('all')

    def decision_plot(self, save=False, fig_path=None):
        if not save:
            shap.decision_plot(self.shap_values.base_values[0], self.shap_values.values)
        else:
            shap.decision_plot(self.shap_values.base_values[0], self.shap_values.values, show=False)
            if fig_path is None:
                fig_path = "shap_decision.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')

        plt.close('all')

    def force_plot(self, idx, save=False, fig_path=None):
        expected = self.shap_values.base_values
        if not save:
            shap.force_plot(expected[idx], self.shap_values.values[idx]).matplotlib((20, 20), True, 0)
        else:
            shap.force_plot(expected[idx], self.shap_values.values[idx]).matplotlib((20, 20), False, 0)
            if fig_path is None:
                fig_path = "shap_force.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')

        plt.close('all')

# Maybe double ML
# make sure that finding is reliable, train and test
# evaluation routines in terms of accuracy/AUC.
# select features based on training set, train classifier on these features only.

# validate on test set both original with all and with less and see if AUC matches
