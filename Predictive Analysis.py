import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

''' 
DATA PREPARATION
'''

df_purchase = pd.read_csv('Data/purchase data.csv')

# Import trained models
scaler = pickle.load(open('Pickle Files/scaler.pickle', 'rb'))
pca = pickle.load(open('Pickle Files/pca.pickle', 'rb'))
kmeans_pca = pickle.load(open('Pickle Files/kmeans_pca.pickle', 'rb'))

# Standardization
features = df_purchase[['Sex', 'Marital status', 'Age', 'Education', 'Income', 'Occupation', 'Settlement size']]
df_purchase_segm_std = scaler.transform(features)

# Apply PCA to the new demographic data which tells us where each person sits
# in each component and allows us to segment them with PCA (the scores)
df_purchase_segm_pca = pca.transform(df_purchase_segm_std)

# K-means PCA - We are now going to find the clusters in which each person sits.
purchase_segm_kmeans_pca = kmeans_pca.predict(df_purchase_segm_pca)
df_purchase_predictors = df_purchase.copy()
df_purchase_predictors['Segment'] = purchase_segm_kmeans_pca

# Create dummy variables
segment_dummies = pd.get_dummies(purchase_segm_kmeans_pca, prefix='Segment', prefix_sep='_')
df_purchase_predictors = pd.concat([df_purchase_predictors, segment_dummies], axis=1)

df_pa = df_purchase_predictors

'''
PURCHASE PROBABILITY MODEL
'''

