'''
Segmentation Dataset - 2000 individuals that make up a sample representative of a population
Data taken from FMCG store
Data has been preprocessed
'''

import pandas as pd
import numpy as np
import missingno as msno
import pandas_profiling as pp
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()



# Import Data and set column 0 to index
df = pd.read_csv('Data/segmentation data.csv', index_col=0)

# Check for null values
msno.matrix(df)

# Explore Data
df.head()
desc = df.describe()
profile = pp.ProfileReport(df, title="Segmentation_Dataset Profile")
profile.to_file("Segmentation_Dataset Profile.html")

# Correlation Estimate - The linear dependency between variables
df_corr = df.corr()     # Pearson Correlation is used here. The Pearson Correlation

# Heatmap
plt.figure(figsize=(12, 9))
s = sns.heatmap(df_corr,
                annot=True,
                cmap='RdBu',
                vmin=-1,
                vmax=1)
s.set_yticklabels(s.get_yticklabels(), rotation=0, fontsize=12)
s.set_xticklabels(s.get_xticklabels(), rotation=45, fontsize=12)
plt.title('Correlation Heatmap')
plt.show()

# Scatterplot of age against income
plt.figure(figsize=(12, 9))
plt.scatter(df.iloc[:, 2], df.iloc[:, 4])
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Age vs. Income Scatterplot')

'''
Now that the dataset has been preprocessed and analysed to a certain extent, we need to prepare the data
for statistical analysis. This can be either Normalization/Standardization.

Our segmentation models will be based on similarity metrics of a consumer. However we are not able to compare
age against salary as the magnitudes will differ greatly.
As a result, we will either standardize or normalize to get the value within a known range.
Normalization will give you a value between 0-1
Standardization will have an arbitrary range with mean of 0 and SD of 1.
'''

scaler = StandardScaler()   # scale is now an INSTANCE of StandardScaler
segmentation_std = scaler.fit_transform(df)

'''
There are two types of clustering:
    - Hierarchical
    - Flat

An example of Hierarchical is the animal kingdom.
The Ward method is used to get the distance between two clusters
    It calculates the average of the square of the distances between clusters.
'''

hier_clust = linkage(segmentation_std, method='ward')

plt.figure(figsize=(12, 9))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Observations')
plt.ylabel('Distance')
dendrogram(hier_clust,
           truncate_mode='level',
           p=5, #only shows the last 5 clusters
           show_leaf_counts=True,
           no_labels=False)
plt.show()

# At the bottom of the dendrogram, we have the 2000 observations along the x axis, and the distances between
# the observations on the y-axis.

'''
K-Means clustering is a flat clustering algorithm.
1) k = the number of clusters we are looking for.
2) Specifying cluster seeds (Starting centroids which are decided by the data scientist as a starting point)
3) Calculate the centroid or geometrical center of the clusters.

K-means doesn't tell us how many clusters there are. It only finds the distance between points.
To get the number of clusters, we select a number of different cluster values.
The Within Cluster Sum of Squares is then calculated (distance between an observation and a centroid)

Issues:
    - The squared euclidean distance is quite sentitive to outliers. 
        -To solve this we can use the K-median clustering, but this is more computationally expensive.
    - We choose the number of clusters.
    - K-means enforces spherical clusters. If we have more elongated data, it won't be as good.
'''

wcss = []  # Within Cluster Sum of Squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(segmentation_std)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 8))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('K-means Cluster')
plt.show()

# Calculate the change in slope to see which cluster is at "ELBOW"
slope_df = pd.DataFrame(data=wcss, columns={'WCSS'})
slope_df['Cluster Size'] = range(1,11)

slopes = []
for i in range(len(slope_df['Cluster Size'])-1):
    slopes.append((slope_df['WCSS'][i+1]-slope_df['WCSS'][i])/(slope_df['Cluster Size'][i+1]-slope_df['Cluster Size'][i]))

# Percentage Change
perc_change = pd.DataFrame(slopes).pct_change()

# Plot
plt.figure(figsize=(10, 8))
plt.plot(range(1, 10), abs(perc_change), marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('Slopes')
plt.title('K-means Cluster')
plt.show()

# Extract the number of clusters based on biggest change in gradients
num_clusters = slope_df['Cluster Size'][np.argmin(perc_change[0])]

# Based on the above, we determine that 4 clusters gives us the best results.
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
kmeans.fit(segmentation_std)
# We now have our clusters and we are going to analyse the data based on these clusters.

'''
RESULTS
'''

df_segm_kmeans = df.copy()
df_segm_kmeans['Segment K-means'] = kmeans.labels_  # This gives us the predicted clusters from the algorithm

# To get some insight we are going to calculate the mean value of each feature by cluster
df_segm_analysis = df_segm_kmeans.groupby(['Segment K-means']).mean()

# Get a count of each group and what proportion that group represents.
# The 'Sex' column needs to be included in this case so that we have a column to count once we have grouped.
df_segm_analysis['Num Obs'] = df_segm_kmeans[['Segment K-means', 'Sex']].groupby(['Segment K-means']).count()
df_segm_analysis['Proportion Obs'] = df_segm_analysis['Num Obs'] / df_segm_analysis['Num Obs'].sum()
df_segm_analysis

df_segm_analysis = df_segm_analysis.rename({0:'Well Off',
                         1:'Fewer Opportunities',
                         2:'Career Focused',
                         3:'Standard'})

# Now plot the data with the new labels
df_segm_kmeans['Labels'] = df_segm_kmeans['Segment K-means'].map({0:'Well Off',
                         1:'Fewer Opportunities',
                         2:'Career Focused',
                         3:'Standard'})

x_axis = df_segm_kmeans['Age']
y_axis = df_segm_kmeans['Income']
plt.figure(figsize=(10, 8))
sns.scatterplot(x_axis, y_axis, hue=df_segm_kmeans['Labels'], palette=['g', 'r', 'c', 'm'])
plt.title('Segmentation Analysis')
plt.show()

'''
PCA - Principal Component Analysis
PCA is the act of reducing dimensionality. This is done by taking all features, and plotting them against
each other, then using linear algebra to find the plane which best fits the data.
'''

pca = PCA()
pca.fit(segmentation_std)  # The number of components is equivalent to the number of features.
expl_vals = pca.explained_variance_ratio_  # PCA orders the components based on how much of the variance of the data is explained /
                               # by each component.
# At this point we are going to want to decrease the number of components we are looking at, whilst preserving
# variance of the data. Therefore we would select the components from most explanatory to least explanatory.

# We can plot the explained variance on a line chart to see how much of the data is explained.
# A rule of thumb states that 60-80% of the variance should be kept to preserve variance as well decrease dimensionality
# Line Chart of Cum Sum of Explained Variance
plt.figure(figsize=(12, 9))
plt.plot(range(1, 8), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
plt.title('Explained Variance by Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')

# Bar chart of Explained variance
plt.figure(figsize=(12, 9))
plt.bar(range(1, 8), pca.explained_variance_ratio_)
plt.title('Explained Variance by Components')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')

# Based on the line graph we can see that the first 3 components explain enough of the variance.
pca = PCA(n_components=3)
pca.fit(segmentation_std)

'''
PCA RESULTS
'''

comps = pca.components_   # Gives us the loadings of the components. The loadings are the correlation between the original
                # variable and the component

df_pca_comp = pd.DataFrame(data=pca.components_,
                           columns=df.columns.values,
                           index=['Component 1', 'Component 2', 'Component 3'])
df_pca_comp

sns.heatmap(df_pca_comp,
            vmin=-1,
            vmax=1,
            cmap='RdBu',
            annot=True)
plt.yticks([0, 1, 2],
           ['Components 1', 'Component 2', 'Component 3'],
           rotation=45,
           fontsize=9)

# We now need to know how each component relates to the original features.
scores_pca = pca.transform(segmentation_std)

'''
K-means clustering with PCA
'''

wcss = []  # Within Cluster Sum of Squares
for i in range(1, 11):
    kmeans_pca = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans_pca.fit(scores_pca)
    wcss.append(kmeans_pca.inertia_)

plt.figure(figsize=(10, 8))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('K-means Cluster')
plt.show()

# Calculate the change in slope to see which cluster is at "ELBOW"
slope_df = pd.DataFrame(data=wcss, columns={'WCSS'})
slope_df['Cluster Size'] = range(1,11)

slopes = []
for i in range(len(slope_df['Cluster Size'])-1):
    slopes.append((slope_df['WCSS'][i+1]-slope_df['WCSS'][i])/(slope_df['Cluster Size'][i+1]-slope_df['Cluster Size'][i]))

# Percentage Change
perc_change = pd.DataFrame(slopes).pct_change()

# Plot
plt.figure(figsize=(10, 8))
plt.plot(range(1, 10), abs(perc_change), marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('Slopes')
plt.title('K-means Cluster')
plt.show()

# Extract the number of clusters based on biggest change in gradients
num_clusters = slope_df['Cluster Size'][np.argmin(perc_change[0])]

# Based on the above, we determine that 4 clusters gives us the best results.
kmeans_pca = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
kmeans_pca.fit(scores_pca)
# We now have the labels for each cluster from our pca dataset.

'''
RESULTS of kmeans_pca
'''

df_segm_kmeans_pca = pd.concat([df.reset_index(drop=True), pd.DataFrame(scores_pca)], axis=1)
df_segm_kmeans_pca.columns.values[-3:] = ['Component 1', 'Component 2', 'Component 3']
df_segm_kmeans_pca['Segment K-means PCA'] = kmeans_pca.labels_  # This gives us the predicted clusters from the algorithm

# To get some insight we are going to calculate the mean value of each feature by cluster
df_kmeans_pca_analysis = df_segm_kmeans_pca.groupby(['Segment K-means PCA']).mean()

# Get a count of each group and what proportion that group represents.
# The 'Sex' column needs to be included in this case so that we have a column to count once we have grouped.
df_kmeans_pca_analysis['Num Obs'] = df_segm_kmeans_pca[['Segment K-means PCA', 'Sex']].groupby(['Segment K-means PCA']).count()
df_kmeans_pca_analysis['Proportion Obs'] = df_kmeans_pca_analysis['Num Obs'] / df_kmeans_pca_analysis['Num Obs'].sum()
df_kmeans_pca_analysis

df_kmeans_pca_analysis = df_kmeans_pca_analysis.rename({0:'Well Off',
                         1:'Fewer Opportunities',
                         2:'Standard',
                         3:'Career Focused'})

# Now plot the data with the new labels
df_segm_kmeans_pca['Labels'] = df_segm_kmeans_pca['Segment K-means PCA'].map({0:'Well Off',
                         1:'Fewer Opportunities',
                         2:'Standard',
                         3:'Career Focused'})

x_axis = df_segm_kmeans_pca['Component 2']
y_axis = df_segm_kmeans_pca['Component 1']
plt.figure(figsize=(10, 8))
sns.scatterplot(x_axis, y_axis, hue=df_segm_kmeans_pca['Labels'], palette=['g', 'r', 'c', 'm'])
plt.title('Segmentation Analysis')
plt.show()

'''
DATA EXPORT
'''

pickle.dump(scaler, open('scaler.pickle', 'wb'))
pickle.dump(pca, open('pca.pickle', 'wb'))
pickle.dump(kmeans_pca, open('kmeans_pca.pickle', 'wb'))
