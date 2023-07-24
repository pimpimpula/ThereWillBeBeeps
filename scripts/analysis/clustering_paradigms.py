import pandas as pd
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt

# read in data and remove subjects with nans
# data = pd.read_pickle("p50_data.pkl")
data = pd.read_pickle("participants_psychometric_curves.pkl")
p50_data = data.drop_duplicates("distance_p50")

df = p50_data.set_index(["paradigm", "pred"])[["participant", "distance_p50"]]
table = df.pivot_table(index=df.index, columns="participant", values="distance_p50")
table.index = [", ".join(x) for x in table.index]
table = table.dropna(axis=1)

fig_dir = '/Users/nadou/Desktop/Figures/'

# dimensionality reduction
pca = PCA()
pcs = pca.fit_transform(table)
print(pca.explained_variance_ratio_.cumsum())

# Scatter plot for the first two principal components
plt.figure(figsize=(7, 7))
plt.scatter(pcs[:, 0], pcs[:, 1], edgecolors='k', s=50)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Scatter Plot')

# Annotate the points with the paradigm and prediction labels
for i, txt in enumerate(table.index):
    plt.annotate(txt, (pcs[i, 0], pcs[i, 1]), fontsize=9, rotation=20, ha='right')

plt.tight_layout()
plt.savefig(f'{fig_dir}PCA_scatter_plot.svg')
plt.show()


#%%
plt.figure(figsize=(10, 7))
# Linking clusters by over all performance
selected_data = pcs[:, :4]  # if you don't want to reduce dimensions:  # table
# use cosine so that you don't worry about mean shifts in data
clusters = shc.linkage(selected_data, method="average", metric="cosine")
shc.dendrogram(Z=clusters, labels=table.index)
plt.xticks(rotation=30, ha="right")
plt.title("Paradigm Dendrogram")
plt.tight_layout()
plt.savefig(f'{fig_dir}paradigm_clustering.svg')
plt.show()
