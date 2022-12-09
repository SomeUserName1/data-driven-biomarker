from abc import ABC
from enum import Enum

import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost
import numpy as np
from sklearn.feature_selection import RFECV, VarianceThreshold, SelectFromModel, SequentialFeatureSelector
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegressionCV, LassoCV

class ModelType(Enum):
    SIMPLE = 0
    TREE = 1
    DEEP = 2
    CLUSTER = 3

class FeatureAnalyzer(ABC):
    def __init__(self, df, samples, model, predict_fn, predictors, 
                 predicteds, model_type=ModelType.SIMPLE, *clust_params):
        self.model_type = model_type
        self.df = df
        self.samples = samples
        self.model = model
        self.predict_fn = predict_fn
        self.predictors = predictors
        self.predicteds = predicteds

        if model_type != ModelType.CLUSTER:
            # compute the SHAP values for the model
            self.explainer = shap.Explainer(predict_fn,
                                            self.df[predictors].iloc[samples])
            self.shap_values = self.explainer(self.df[predictors])
        else:
            # get detected cluster labels
            y = model.fit(self.df[predictors], clust_params).labels_
            y = label_binarize(y, classes=range(len(y)))
            # fit a classifier
            clf = xgboost.XGBClassifier()
            clf.fit(self.df[predictors], y)
            # compute the shap values per class and feature
            self.explainer = shap.TreeExplainer(clf)
            self.shap_values = self.explainer(self.df[predictors])
        # maybe add sth to deal with dimensionality reduction


    def remove_zero_variance_feats(self):
        selector = VarianceThreshold()
        selector.fit(self.df[self.predicts])
        self.predicts = selector.get_feature_names_out()

    def get_max_importance(self):
        return np.max(self.shap_values, axis=2)

    # Feature importance/coefficient- free methods
    def backwards_select(self):
        backward = SequentialFeatureSelector(self.model, n_features_to_select="auto", direction="backward", tol=0.05)
        backward.fit(self.df[self.predictors])
        self.predictors = backward.get_feature_names_out()

    def forwards_select(self):
        forward = SequentialFeatureSelector(self.model, n_features_to_select="auto", direction="forward", tol=0.05)
        forward.fit(self.df[self.predictors])
        self.predictors = forward.get_feature_names_out()

    # logit for classification, lasso for regression feature elimination
    def logit_select(self):
        logit = LogisticRegressionCV(penalty='l1', solver='liblinear', n_jobs=-1)
        logit.fit(self.df[self.predictors], self.df[self.predicteds])
        selector = SelectFromModel(logit, prefit=True)
        selector.fit(self.df[self.predictors])
        self.predicts = selector.get_feature_names_out()

    def lasso_select(self):
        lasso = LassoCV(n_jobs=-1)
        lasso.fit(self.df[self.predictors], self.df[self.predicteds])
        selector = SelectFromModel(lasso, prefit=True)
        selector.fit(self.df[self.predictors])
        self.predicts = selector.get_feature_names_out()

    # Recursive feature eleimination & simple feature importance based select
    def recursive_feature_elimination(self):
        rfecv = RFECV(estimator=self.model, n_jobs=-1, importance_getter=self.get_max_importance)
        rfecv.fit(self.df[self.predictors], self.df[self.predicteds])
        self.predicts = rfecv.get_feature_names_out()

    def shap_select(self):
        selector = SelectFromModel(self.model, prefit=True, importance_getter=self.get_max_importance)
        selector.fit(self.df[self.predictors])
        self.predicts = selector.get_feature_names_out()

    def correlation_matrix_plot(self):
        plt.figure(figsize=(20,20))
        sns.heatmap(self.df.corr(), annot=True, cmap="RdYlGn")

    def partial_dependence_plot(self, col, index):
    # make a standard partial dependence plot
        shap.partial_dependence_plot(
            col, self.predict_fn, self.df[self.predictors].iloc[self.samples],
            model_expected_value=True, feature_expected_value=True,
            ice=False,
            shap_values=self.shap_values[index:index+1,:]
        )

    def scatter_plot(self, col):
    # Shap value scatter plot
        shap.plots.scatter(self.shap_values[:, col], color=self.shap_values)

    def bar_plot(self, feature_correl=True, clustering_cutoff=0.8):
        # Correlated features shown by dendrogram
        clustering = shap.utils.hclust(self.df[self.predictors],
                                       self.df[self.predicteds])

        # Mean shap value bar plot
        shap.plots.bar(self.shap_values, clustering=clustering,
                       clustering_cutoff=clustering_cutoff)
        shap.plots.bar(self.shap_values.abs.max(0), clustering=clustering,
                       clustering_cutoff=clustering_cutoff)
        shap.plots.bar(self.shap_values.abs.min(0), clustering=clustering,
                       clustering_cutoff=clustering_cutoff)


    def waterfall_plot(self, index):
    # the waterfall_plot shows how we get from shap_values.base_values to model.predict(X)[sample_ind]
        shap.plots.waterfall(self.shap_values[index], max_display=14)

    def beeswarm_plot(self):
    # Beeswarm plot summarizes entire distribution of SHAP values f.a. features
        shap.plots.beeswarm(self.shap_values)

    def heatmap_plot(self):
    # Heatmaps show value distribution vs. shap value per feature
        shap.plots.heatmap(self.shap_values)

    def decision_plot(self):
        if self.model_type not in [ModelType.TREE, ModelType.CLUSTER]:
            print("Decision plots can only be generated for tree models!")
            return

        shap.decision_plot(self.explainer.expected_value, self.shap_values, self.df[self.predictors])

    def force_plot(self):
        if self.model_type not in [ModelType.TREE, ModelType.CLUSTER]:
            print("Decision plots can only be generated for tree models!")
            return

        shap.force_plot(self.explainer.expected_value, self.shap_values, self.df[self.predictors], matplotlib=True) 

