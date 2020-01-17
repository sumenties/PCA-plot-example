## Make sure to install and import necessary modules first

import pandas as pd
import numpy as np
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt

# Loading the data
# Note: This is already a preprocessed data.

data = pd.read_csv('./pca_example_dataset.csv')

## In this example, the data is in a data frame called data.
## Columns represent different samples (i.e. cells) that may have been under specific conditions to alter gene expression.
## Rows represent genes from these different samples.

# Checking the head (First 5 columns) and shape of the data.
print(data.head())
print(data.shape)

# Drop geneid before applying PCA as it has no significance in statistics.
d = data.drop("geneid", 1)
d.head() # checking if geneid is removed or not.
d.shape

# First center and scale the data
scaled_data = preprocessing.scale(d.T)

pca = PCA()  # create a PCA object
pca.fit(scaled_data)  # do the statistics
pca_data = pca.transform(scaled_data)  # get PCA coordinates for scaled_data

# Drawing a scree plot and a PCA plot:

# The following code constructs the Scree plot
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]

plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

# the following code creates PC1 and PC2 in a 2D PCA plot:
pca_df = pd.DataFrame(pca_data, columns=labels)

plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title('My PCA Graph')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))

for sample in pca_df.index:
    plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))

plt.show()

# Determine which genes had the biggest influence on PC1:

## get the name of the top 10 measurements (genes) that contribute
## most to pc1.
## first, get the loading scores
loading_scores = pd.Series(pca.components_[0])

## now sort the loading scores based on their magnitude
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)

# get the names of the top 10 genes
top_10_genes = sorted_loading_scores[0:10].index.values
print(top_10_genes)

