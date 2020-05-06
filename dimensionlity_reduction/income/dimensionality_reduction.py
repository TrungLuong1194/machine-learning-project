# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('income_evaluation.csv')

# Remove Nan value
dataset = dataset.replace(' ?', np.nan).dropna()

# Encoding target field (feature 'income')
dataset = dataset.replace(' >50K', 1)
dataset = dataset.replace(' <=50K', 0)

# Remove feature 'fnlwgt'
dataset = dataset.drop('fnlwgt', axis=1)

# Setting features, targets
target = dataset['income']
feature = dataset.drop('income', axis=1)

# Categorizing variables
feature_dummies = pd.get_dummies(feature)

# Setting X, y
X = feature_dummies.values
y = target.values

# Feature scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

# -----------------------------------------------------------------------------
# Dimensionality Reduction by PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=None)
X_pca = pca.fit_transform(X_scaled)

# Find best value for n_components: d
cumsum = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(20,10))
plt.bar(range(1, 104), pca.explained_variance_ratio_, alpha=0.5, align='center',
        label='invididual explained variance')
plt.step(range(1, 104), cumsum, where='mid',
         label='cummulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.show()

d = np.argmax(cumsum >= 0.8) + 1

# Rebuilding a model with best parameters
pca = PCA(n_components=d)
X_train_pca = pca.fit_transform(X_scaled)

# Principal component
principal_component = pca.components_

plt.figure(figsize=(20,10))
plt.matshow(pca.components_, cmap='viridis')
plt.yticks(range(0, d, 3))
plt.colorbar()
plt.xticks(range(0, 103, 10), rotation=60, ha='left')
plt.xlabel("Feature")
plt.ylabel("Principal components")
plt.show()

# -----------------------------------------------------------------------------
# Dimensionality Reduction by t-SNE

from sklearn.manifold import TSNE

tsne = TSNE(random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

colors = ["#FF0000", "#00FF00"]
plt.figure(figsize=(20,20))
plt.xlim(X_tsne[:, 0].min(), X_tsne[:, 0].max() + 1)
plt.ylim(X_tsne[:, 1].min(), X_tsne[:, 1].max() + 1)
for i in range(len(X_scaled)):
    plt.text(X_tsne[i, 0], X_tsne[i, 1], str(y[i]),
             color = colors[y[i]],
             fontdict={'weight': 'bold', 'size': 9})
plt.xlabel("t-SNE feature 0")
plt.ylabel("t-SNE feature 1")
plt.show()