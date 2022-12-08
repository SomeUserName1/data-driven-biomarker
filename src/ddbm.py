from abc import ABC

import shap

class FeatureAnalyzer(ABC):
    def __init__(self, df, samples, model, predict_fn, predictors, predicteds):
        self.df = df
        self.samples = samples
        self.model = model
        self.predict_fn = predict_fn
        self.predictors = predictors
        self.predicteds = predicteds
        # compute the SHAP values for the linear model
        self.explainer = shap.Explainer(predict_fn,
                                        self.df[predictors].iloc(samples))
        self.shap_values = self.explainer(self.df[predictors])


    def partial_dependence_plot(self, col, index):
    # make a standard partial dependence plot
        shap.partial_dependence_plot(
            col, self.predict_fn, self.df[self.predictors].iloc(self.samples),
            model_expected_value=True, feature_expected_value=True,
            ice=False,
            shap_values=self.shap_values[index:index+1,:]
        )

    def scatter_plot(self, col):
    # Shap value scatter plot
        shap.plots.scatter(self.shap_values[:, col], color=self.shap_values)

    def bar_plot(self, feature_correl=True, clustering_cutoff=0.8):
    # Mean shap value bar plot
        shap.plots.bar(self.shap_values)
        # Correlated features shown by dendrogram
        clustering = shap.utils.hclust(self.df[self.predictors],
                                       self.df[self.predicteds])

        shap.plots.bar(self.shap_values, clustering=clustering,
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
        # TODO multioutput
        shap.decision_plot(self.explainer.expected_value, self.shap_values, self.df[self.predictors])

    def force_plot(self):
        shap.force_plot(self.explainer.expected_value, self.shap_values, self.df[self.predictors], matplotlib=True) 
