#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 11:33:31 2022

@author: enricoferrea
"""

# import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
from sklearn import datasets
from scipy.stats import zscore

from utils import confidence_ellipse

from matplotlib.patches import Rectangle

#%% Load questionnaires

df_UPDRS_OFF = pd.read_excel('/home/enricoferrea/Desktop/templates/MDS-UPDRS III_med_OFF.xlsx')

df_UPDRS_ON = pd.read_excel('/home/enricoferrea/Desktop/templates/MDS-UPDRS III_med_ON.xlsx')    

df_PDQ39 = pd.read_excel('/home/enricoferrea/Desktop/templates/PDQ39.xlsx')  

df_MOCA = pd.read_excel('/home/enricoferrea/Desktop/templates/MoCA.xlsx') 

df_Beck = pd.read_excel('/home/enricoferrea/Desktop/templates/BDI Version 1.xlsx')


# Drop last three colums UDPRS
df_UPDRS_OFF = df_UPDRS_OFF.drop(columns = ["MDS-UPDRS_Dyskinesien","MDS-UPDRS_Einfluss","MDS-H&Y"])
df_UPDRS_ON = df_UPDRS_ON.drop(columns = ["MDS-UPDRS_Dyskinesien","MDS-UPDRS_Einfluss","MDS-H&Y"])

df_UPDRS_OFF = df_UPDRS_OFF.sort_values(by='PatientID',ascending = True)
df_UPDRS_ON = df_UPDRS_ON.sort_values(by='PatientID',ascending = True)  

df_PDQ39 = df_PDQ39.sort_values(by='PatientID',ascending = True)
df_MOCA = df_MOCA.sort_values(by='PatientID',ascending = True)
df_Beck = df_Beck.sort_values(by='PatientID',ascending = True)
 

#df_data = df_data.iloc[0:10, 0:-1]

# df_data = pd.merge(df_PDQ39, df_UPDRS, on="PatientID")

# df_data = pd.merge(df_data, df_MOCA , on="PatientID")

# df_data = pd.merge(df_data, df_Beck, on="PatientID")



df_data = df_Beck 
#df_data = df_UPDRS_OFF
#%% MErge all questionnaires

df_plist = pd.read_excel('/home/enricoferrea/Desktop/Questionnaires/2011-2021 Alles.xlsx')
df_plist['PatientID'] = df_plist.index+2
df_all = df_PDQ39
df_all = pd.merge(df_UPDRS_OFF , df_all, on="PatientID",how="outer")
df_all = pd.merge(df_MOCA , df_all, on="PatientID",how="outer")
df_all = pd.merge(df_Beck , df_all, on="PatientID",how="outer")
df_all = pd.merge(df_plist , df_all, on="PatientID",how="inner")
df_all  = df_all.sort_values(by='PatientID',ascending = True) 
# df_data = df
#df_data = df_UPDRS
# df.head()
# dtypes = df.dtypes
# df.info()
IDlist = df_all[['PatientID','OP Datum']] 
IDlist['OP Datum'] = IDlist['OP Datum'].str.slice(0, 10)
#%% add missing values


missing = df_data.isna()
print("missing values before subst:", df_data.isna().sum().sum())

df_data = df_data.fillna(df_data.median())

missing = df_data.isna()
print("missing values after subst:", df_data.isna().sum().sum())

df_data = df_data.dropna(axis = 1)

missing = df_data.isna()
print("missing values after col removal:", df_data.isna().sum().sum())

#%% Calculate PCA and plot

n_components = 6

# Standardizing the features
#x = StandardScaler().fit_transform(df_data.drop(columns = ['PatientID']))
#x = df_data.iloc[:, 1:-1]
x = df_data.drop(columns = ['PatientID'])
x = x.apply(zscore)

pca = PCA(n_components = n_components)
principalComponents = pca.fit_transform(x)
print(pca.explained_variance_ratio_)

#%% Explore number of components

plt.figure()
plt.plot(range(1,n_components+1),pca.explained_variance_ratio_.cumsum(),marker = 'o', linestyle = '--')
plt.title("Explained Variance by components")
plt.xlabel("number of componens")
plt.ylabel('Cumulative Explained Variance')
         

#%%

principalDf = pd.DataFrame(data = principalComponents[:,0:3]
             , columns = ['PCA1', 'PCA2','PCA3'])

finalDf = pd.concat([principalDf, df_data], axis = 1)


PDQ_Total= df_data.iloc[:, 1:-1].sum(axis=1)
thresholds = np.multiply(PDQ_Total > 10,1)
#%% plot results 2D
colormap = np.array(['r', 'b', 'g','k'])
colormap = np.array(['tab:blue', 'tab:orange', 'tab:green'])
fig = plt.subplots()
plt.scatter(finalDf["PCA1"], finalDf["PCA2"],c =colormap[thresholds],  alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA space and threshold clustering')
plt.show()
#


#%% plot results 3D
colormap = np.array(['r', 'b', 'g','k'])
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
colormap = np.array(['tab:blue', 'tab:orange', 'tab:green'])

ax.scatter(finalDf["PCA1"], finalDf["PCA2"],finalDf["PCA3"],c =colormap[thresholds])

ax.set_xlabel('PC1 Label')
ax.set_ylabel('PC2 Label')
ax.set_zlabel('PC3 Label')

plt.show()



#%% apply TSNE

from sklearn.manifold import TSNE

x_embedded = TSNE(n_components=3, learning_rate='auto',init='random', perplexity=5).fit_transform(x)

tsneDf = pd.DataFrame(data = x_embedded 
             , columns = ['tsne1', 'tsne2','tsne3'])

finalDf = pd.concat([tsneDf, finalDf], axis = 1)

#%%2D plot
fig = plt.subplots()
plt.scatter(finalDf["tsne1"], finalDf["tsne2"], c = thresholds, alpha=0.5)
plt.show()

#%%3D plot


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(finalDf["tsne1"], finalDf["tsne2"],finalDf["tsne3"],c = thresholds)

ax.set_xlabel('T1 Label')
ax.set_ylabel('T2 Label')
ax.set_zlabel('T3 Label')

plt.show()

#%% calculate kmeans PCA
random_state = 170


X = finalDf[["PCA1","PCA2","PCA3"]].to_numpy()

y_predPCA = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)


#%% Do prediction on all features
random_state = 170


#X = StandardScaler().fit_transform(df_data.drop(columns = ['PatientID']))
X = df_data.drop(columns = ['PatientID'])

X = X.apply(zscore)

kmeans = KMeans(n_clusters=3, random_state=random_state).fit(X)

y = kmeans.predict(X)


nclass_0 = np.sum(y == 0)/np.size(y,0)

#%% plot results 3D with labes kMeans



fig = plt.figure()
ax = fig.add_subplot(projection='3d')
colormap = np.array(['r', 'g', 'b','k'])
colormap = np.array(['tab:blue', 'tab:orange', 'tab:green'])
colormap = np.array([ 'green','fuchsia','deepskyblue' ])

scatter = ax.scatter(finalDf["PCA1"], finalDf["PCA2"],finalDf["PCA3"],c =colormap[y],cmap='viridis',label= y)
ax.set_title("K-means clustering")
# extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
# ax.legend([extra,y], ("My explanatory text", "0-10", "10-100"))
ax.set_xlabel('PC1 Label')
ax.set_ylabel('PC2 Label')
ax.set_zlabel('PC3 Label')



plt.show()
#%% plot results 2D
colormap = np.array(['r', 'g', 'b','k'])
colormap = np.array(['tab:green', 'tab:pink','tab:blue'])
fig = plt.subplots()
plt.scatter(finalDf["PCA1"], finalDf["PCA2"],c= colormap[y],alpha=0.5)
#plt.legend()
plt.grid(True)
plt.show()


#%% find features inmportance as supervised classification problem 
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
fig = plt.subplots()
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(30).plot(kind='barh')
plt.show()
#%%
# #%% Use Shaplhey values



# import xgboost as xgb
# model = xgb.XGBClassifier(n_estimators=20, max_depth=2, base_score=0.5,
#                         objective='binary:logistic', random_state=42)
# model.fit(X, y)


# fig = plt.figure(figsize = (16, 12))
# title = fig.suptitle("Default Feature Importances from XGBoost", fontsize=14)

# ax1 = fig.add_subplot(2,2, 1)
# xgb.plot_importance(model, importance_type='weight', ax=ax1)
# t=ax1.set_title("Feature Importance - Feature Weight")

# ax2 = fig.add_subplot(2,2, 2)
# xgb.plot_importance(model, importance_type='gain', ax=ax2)
# t=ax2.set_title("Feature Importance - Split Mean Gain")

# ax3 = fig.add_subplot(2,2, 3)
# xgb.plot_importance(model, importance_type='cover', ax=ax3)
# t=ax3.set_title("Feature Importance - Sample Coverage")





# #%%
# # import shap library
# import shap

# # explain the model's predictions using SHAP
# explainer = shap.Explainer(model,X)
# shap_values = explainer(X)
# fig = plt.subplots()
# shap.summary_plot(shap_values[:,:,2], X, plot_type="bar")

# shap.plots.scatter(shap_values)
# shap.plots.waterfall(explainer.shap_values(X), max_display=10, show=True)
# fig = plt.subplots()
# shap.plots.waterfall(shap_values[1,:,0])
# fig = plt.subplots()
# shap.plots.beeswarm(shap_values[:,:,0])
# fig = plt.subplots()
# shap.plots.beeswarm(shap_values[:,:,1])
# fig = plt.subplots()
# shap.plots.scatter(shap_values[:,:,1])
# shap.plots.waterfall(shap_values[:,:,0])
# shap.waterfall_plot(explainer.base_values[0],shap_values[0], X[0])
# # fig = plt.subplots()
# # shap.initjs()
# #shap.plots.force(explainer.expected_value, shap_values=shap_values[1], features=X.iloc[1], matplotlib = True)
#%%

import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X,approximate=True)
fig = plt.subplots()
shap.summary_plot(shap_values, X, plot_type="bar")
fig = plt.subplots()
shap.summary_plot(shap_values[1], X)



#%%
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X,approximate=True)
fig = plt.subplots()
shap.summary_plot(shap_values, X, plot_type="bar")
fig = plt.subplots()
shap.summary_plot(shap_values[1], X)
#%% plot results 2D
colormap = np.array(['r', 'g', 'b'])
fig = plt.subplots()
plt.scatter(finalDf["PCA1"], finalDf["PCA2"],c= colormap[y],  alpha=0.5)
plt.show()
#%% use sns to plot
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")


f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.scatterplot(x="PCA1", y="PCA2",
                c = y,
                hue = y,
                palette="viridis",
                sizes=(1, 8), linewidth=0,
                data=finalDf, ax=ax)

#%% plot results 3D with labes kMeans



fig = plt.figure()
ax = fig.add_subplot(projection='3d')
colormap = np.array(['r', 'g', 'b'])

ax.scatter(finalDf["PCA1"], finalDf["PCA2"],finalDf["PCA3"],c =colormap[y])

ax.set_xlabel('PC1 Label')
ax.set_ylabel('PC2 Label')
ax.set_zlabel('PC3 Label')

plt.show()







#%% find left vs right dimensions. Only for UDPRS
df_data = df_UPDRS

df_data["right"] = df_data["MDS-UPDRS_3.3ROE"] +df_data["MDS-UPDRS_3.3RUE"] + df_data["MDS-UPDRS_3.5R"] + df_data["MDS-UPDRS_3.6R"] \
    + df_data["MDS-UPDRS_3.7R"] + df_data["MDS-UPDRS_3.8R"] +  df_data["MDS-UPDRS_3.15R"] +df_data["MDS-UPDRS_3.16R"] + df_data["MDS-UPDRS_3.17ROE"] \
        + df_data["MDS-UPDRS_3.17RUE"] 
        
        
        
df_data["left"] = df_data["MDS-UPDRS_3.3LOE"] +df_data["MDS-UPDRS_3.3LUE"] + df_data["MDS-UPDRS_3.5L"] + df_data["MDS-UPDRS_3.6L"] \
    + df_data["MDS-UPDRS_3.7L"] + df_data["MDS-UPDRS_3.8L"] +  df_data["MDS-UPDRS_3.15L"] +df_data["MDS-UPDRS_3.16L"] + df_data["MDS-UPDRS_3.17LOE"] \
        + df_data["MDS-UPDRS_3.17LUE"] 




#%% plot
prevalence_side_color = np.multiply(df_data["left"] > df_data["right"],1)
prevalence_side = df_data["left"] > df_data["right"]
colormap = np.array(['tab:blue', 'tab:orange', 'tab:green'])
fig,ax = plt.subplots()
ax.scatter(df_data["left"], df_data["right"], c = colormap[prevalence_side_color], alpha=0.5)
ax.axline((0, 0), slope=1, color="black", linestyle=(0, (5, 5)))
ax.axvline(c='grey', lw=1)
ax.axhline(c='grey', lw=1)

confidence_ellipse(df_data["left"][~prevalence_side],  df_data["right"][~prevalence_side], ax, n_std=2,
                   label=r'$1\sigma$', edgecolor='fuchsia',linestyle=':')
confidence_ellipse(df_data["left"][prevalence_side],  df_data["right"][prevalence_side], ax, n_std=2,
                   label=r'$1\sigma$', edgecolor='firebrick',linestyle='--')
ax.axis('equal')
plt.xlabel('Left score')
plt.ylabel('Right score')
plt.title('Left Right dominance')
plt.show()




