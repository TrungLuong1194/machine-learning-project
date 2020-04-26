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

# Splitting the dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

# Feature scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

# -----------------------------------------------------------------------------
# Clustering by K-Means and Elbow method
from sklearn.cluster import KMeans

distortions_km = []
for i in range(1, 11):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(X_train_scaled)
    distortions_km.append(km.inertia_)
    
# Plot the distortion for different values of k
plt.plot(range(1, 11), distortions_km, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

# Rebuilding a model with best parameters
km = KMeans(n_clusters=2,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=0)

y_km = km.fit_predict(X_train_scaled)

# Evaluating method
from sklearn.metrics import silhouette_score

silhouette_score_km = silhouette_score(X_train_scaled, y_km)

# -----------------------------------------------------------------------------
# Clustering by Agglomerative clustering
from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=2,
                             affinity='euclidean',
                             linkage='complete')

y_ac = ac.fit_predict(X_train_scaled)

# Evaluating method
silhouette_score_ac = silhouette_score(X_train_scaled, y_ac)