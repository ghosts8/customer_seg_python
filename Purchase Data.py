import pandas as pd
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

'''
PURCHASE ANALYTICS with DESCRIPTIVE STATS
'''

df_purchase = pd.read_csv('Data/purchase data.csv')

# Check for null values
msno.matrix(df_purchase)

# Explore Data
df_purchase.head()
desc = df_purchase.describe()
profile = pp.ProfileReport(df_purchase, title="Purchase_Dataset Profile")
profile.to_file("Purchase_Dataset Profile.html")

# Correlation Estimate - The linear dependency between variables
df_purchase_corr = df_purchase.corr()     # Pearson Correlation is used here. The Pearson Correlation

# Heatmap
plt.figure(figsize=(12, 9))
s = sns.heatmap(df_purchase_corr,
                annot=True,
                cmap='RdBu',
                vmin=-1,
                vmax=1)
s.set_yticklabels(s.get_yticklabels(), rotation=0, fontsize=12)
s.set_xticklabels(s.get_xticklabels(), rotation=45, fontsize=12)
plt.title('Correlation Heatmap')
plt.show()

# # Scatterplot of age against income
# plt.figure(figsize=(12, 9))
# plt.scatter(df_purchase.iloc[:, 2], df_purchase.iloc[:, 4])
# plt.xlabel('Age')
# plt.ylabel('Income')
# plt.title('Age vs. Income Scatterplot')

'''
Apply previously trained segmentation model
'''

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

# Explore Data
df_purchase_predictors.head()
df_purchase_predictors.describe()
profile = pp.ProfileReport(df_purchase_predictors, title="df_purchase_predictors Profile")
profile.to_file("df_purchase_predictors Profile.html")

# Correlation Estimate - The linear dependency between variables
df_purchase_predictors_corr = df_purchase_predictors.corr()     # Pearson Correlation is used here. The Pearson Correlation

# Heatmap
plt.figure(figsize=(12, 9))
s = sns.heatmap(df_purchase_predictors_corr,
                annot=True,
                cmap='RdBu',
                vmin=-1,
                vmax=1)
s.set_yticklabels(s.get_yticklabels(), rotation=0, fontsize=12)
s.set_xticklabels(s.get_xticklabels(), rotation=45, fontsize=12)
plt.title('Correlation Heatmap')
plt.show()

# Let's have a look at the dataset per customer.
# temp1 tells us how many times a per person has visited the store. Therefore grouping by ID(cust) and
# counting incidence will tell us how many times they have gone to the store. (Incidence is arbitrarily chosen)
temp1 = df_purchase_predictors[['ID', 'Incidence']].groupby(['ID'], as_index=False).count()
temp1 = temp1.set_index('ID')
temp1 = temp1.rename(columns={'Incidence': 'N_Visits'})
temp1.head()

# temp2 tells us how many purchases per person. A purchase doesn't represent how many chocolate bars were bought
# We are summing here because we have values 1/0 per person in the incidence column.
temp2 = df_purchase_predictors[['ID', 'Incidence']].groupby(['ID'], as_index=False).sum()
temp2 = temp2.set_index('ID')
temp2 = temp2.rename(columns={'Incidence': 'N_Purchases'})
temp2.head()

# temp3 tells us how probable someone is to buy a chocolate bar upon visting.
# Eg: The how many times a person makes a purchase when they visit the store
# temp3 = pd.concat([temp1, temp2], axis=1)
temp3 = temp1.join(temp2)
temp3['Purchase Freq'] = temp3['N_Purchases']/temp3['N_Visits']
temp3.head()

# temp4 is which segment each person is in. You can either take the first segment, or the means of segments.
temp4 = df_purchase_predictors[['ID', 'Segment']].groupby(['ID']).first()
temp5 = df_purchase_predictors[['ID', 'Segment']].groupby(['ID']).mean()
(temp5 - temp4).sum()  # Quick check to see they return the same data

# We now join the segment into or previous analysis
df_purchase_descr = temp3.join(temp5)
df_purchase_descr.head()

# This is some segment analysis:
# n_visits_segm gives us how many visits the store is getting per segment
n_visits_segm = df_purchase_descr[['Segment', 'N_Visits']].groupby(['Segment']).sum()
# n_purchases_segm gives us how many purchases are being made per segment
n_purchases_segm = df_purchase_descr[['Segment', 'N_Purchases']].groupby(['Segment']).sum()
# n_people_segm gives us the distinct number of people there are per segment. (N_Purchases was arbitrarily chosen)
n_people_segm = df_purchase_descr[['Segment', 'N_Purchases']].groupby(['Segment']).count()
n_people_segm = n_people_segm.rename(columns={'N_Purchases': 'N_People'})
# Segm_Prop is proportion of people in each segment from the total sample.
n_people_segm['Segm_Prop'] = n_people_segm / len(df_purchase_descr)

segm_analysis = n_visits_segm.join(n_purchases_segm)
segm_analysis['Purchase Freq'] = segm_analysis['N_Purchases'] / segm_analysis['N_Visits']
segm_analysis = segm_analysis.join(n_people_segm)
# segm_analysis_desc = segm_analysis.reset_index().sort_values(by=['Purchase Freq'], ascending=False)

# Subplot 3 barcharts to analyse the different segments and their metrics
fig, ax = plt.subplots(1, 4)
x_axis = segm_analysis.index.values
for i in range(len(segm_analysis.columns.values)-1):
    sns.barplot(x_axis, segm_analysis.iloc[:, i], data=segm_analysis, ax=ax[i])
plt.subplots_adjust(wspace=1)

'''
Purchase Occasion and Purchase Incidence
'''

# This is looking at the mean/average number of Visits, Purchases, and Purchase Frequency per Segment
segments_mean = df_purchase_descr.groupby(['Segment']).mean()
segments_mean = segments_mean.rename(columns={'N_Visits': 'Avg_N_Visits', 'N_Purchases': 'Avg_N_Purchases', 'Purchase Freq': 'Avg_Purchase_Freq'})

# This is calculating the STD of the above
segments_std = df_purchase_descr.groupby(['Segment']).std()
segments_std = segments_std.rename(columns={'N_Visits': 'STD_N_Visits', 'N_Purchases': 'STD_N_Purchases', 'Purchase_Freq': 'STD_Purchase_Freq'})

fig, ax = plt.subplots(1, 3)
x_axis = segments_mean.index.values
x_labels = ['Well-Off', 'Fewer-Opportunities', 'Standard', 'Career-Focused']
for i in range(len(segments_mean.columns.values)):
    g = sns.barplot(x=x_axis, y=segments_mean.iloc[:, i], data=segments_mean, yerr=segments_std.iloc[:, i], ax=ax[i])
    g.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    for j in g.get_xticklabels():
        j.set_rotation(45)
plt.subplots_adjust(wspace=1)

'''
BRAND CHOICE
'''

df_purchase_incidence = df_purchase_predictors[df_purchase_predictors['Incidence'] == 1]

brand_dummies = pd.get_dummies(df_purchase_incidence['Brand'], prefix='Brand', prefix_sep='_')
brand_dummies['Segment'], brand_dummies['ID'] = df_purchase_incidence['Segment'], df_purchase_incidence['ID']

temp = brand_dummies.groupby(['ID'], as_index=True).mean()

avg_brand_per_seg = temp.groupby(['Segment'], as_index=True).mean()
avg_brand_per_seg = avg_brand_per_seg.rename({0:'Well Off',
                         1:'Fewer Opportunities',
                         2:'Standard',
                         3:'Career Focused'})

sns.heatmap(avg_brand_per_seg,
            vmin=0,
            vmax=1,
            cmap='PuBu',
            annot=True)
plt.yticks([0, 1, 2, 3], ['Well Off', 'Fewer Opportunities', 'Standard', 'Career Focused'], rotation=45)
plt.title('Average Brand Choice by Segment')
# The brands were added to the dataset from least expensive to most expensive, from left to right.


def brand_rev_per_seg(brand):
    '''Calculates the revenue for a brand per segment'''
    temp = df_purchase_predictors[df_purchase_predictors['Brand'] == brand]
    temp['Revenue Brand ' + str(brand)] = temp['Price_' + str(brand)] * temp['Quantity']
    segments_brand_revenue[['Segment', 'Revenue Brand ' + str(brand)]] = temp[['Segment', 'Revenue Brand ' + str(brand)]].groupby(['Segment'], as_index=False).sum()


segments_brand_revenue = pd.DataFrame()

# Iterate through each brand
for i in range(1, 6):
    brand_rev_per_seg(i)

# Calculate the total revenue per segment and set the index to segment
segments_brand_revenue.set_index('Segment', inplace=True)
segments_brand_revenue['Totals'] = segments_brand_revenue.sum(axis=1)


segments_brand_revenue['Seg_Proportions'] = segm_analysis['Segm_Prop']
segments_brand_revenue.index = segments_brand_revenue.index.map({0:'Well Off',
                                                                 1:'Fewer-Opportunities',
                                                                 2:'Career-Focused',
                                                                 3:'Standard'})


# Display barplots for revenue in each brand per segment
fig, ax = plt.subplots(1, len(segments_brand_revenue.columns.values))
x_axis = segments_brand_revenue.index.values
x_labels = ['Well-Off', 'Fewer-Opportunities', 'Standard', 'Career-Focused']
for i in range(len(segments_brand_revenue.columns.values)):
    g = sns.barplot(x_axis, segments_brand_revenue.iloc[:, i], data=segments_brand_revenue, ax=ax[i])
    g.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    for j in g.get_xticklabels():
        j.set_rotation(45)
fig.suptitle('Brand Revenue per Segment')
fig.subplots_adjust(wspace=1)

