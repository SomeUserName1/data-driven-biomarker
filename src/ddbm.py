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

    def shap_select(self, shap_max=True):
        fn = self.get_max_importance if shap_max else self.get_mean_importance
        selector = SelectFromModel(self.model, prefit=True, threshold="median",
                                   max_features=int(0.9 * len(self.predictors)),
                                   importance_getter=fn)
        selector.fit(self.df[self.predictors])
        print(selector.get_feature_names_out())
        print(fn(None)[selector.get_support(indices=True)])

    def correlation_matrix_plot(self):
        plt.figure(figsize=(20,20))
        sns.heatmap(self.df.corr(), annot=True, cmap="RdYlGn")
        plt.show()

#    def shap_interaction_values(self):
#        if self.model_type in [ModelType.TREE, ModelType.CLUSTER]:
#            print("Interaction values:")
#            print(self.explainer.shap_interaction_values(self.X))

    def partial_dependence_plot(self, col):
    # make a standard partial dependence plot
        shap.plots.partial_dependence(
            col, self.predict_fn, self.X,
            model_expected_value=True, feature_expected_value=True,
            ice=False
        )

    def scatter_plot(self, col, interaction=False):
    # Shap value scatter plot
        shap.plots.scatter(self.shap_values[:, col], color=self.shap_values)

    def bar_plot(self, feature_correl=True, clustering_cutoff=0.8):
        # Correlated features shown by dendrogram
        clustering = shap.utils.hclust(self.df[self.predictors],
                                       self.df[self.predicteds])
        # Mean shap value bar plot
        shap.plots.bar(self.shap_values, max_display=self.max_display, clustering=clustering,
                       clustering_cutoff=clustering_cutoff)


    def waterfall_plot(self, index):
    # the waterfall_plot shows how we get from shap_values.base_values to
    # model.predict(X)[sample_ind]
        shap.plots.waterfall(self.shap_values[index])

    def beeswarm_plot(self):
    # Beeswarm plot summarizes entire distribution of SHAP values f.a. features
        shap.plots.beeswarm(self.shap_values, max_display=self.max_display)

    def heatmap_plot(self):
    # Heatmaps show value distribution vs. shap value per feature
        shap.plots.heatmap(self.shap_values, max_display=self.max_display)

    def decision_plot(self):
        shap.decision_plot(self.shap_values.base_values[0], self.shap_values.values, max_display=self.max_display)

    def force_plot(self, idx):
        expected = self.shap_values.base_values
        shap.force_plot(expected[idx], self.shap_values.values[idx]).matplotlib((20, 20), True, 0)

# Maybe double ML
# make sure that finding is reliable, train and test
# evaluation routines in terms of accuracy/AUC.
# select features based on training set, train classifier on these features only.

# validate on test set both original with all and with less and see if AUC matches
