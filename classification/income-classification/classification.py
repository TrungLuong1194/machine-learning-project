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
        X, y, test_size=0.15, random_state=0)

# Feature scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------------------------------
# Classification by Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

param_grid_lr = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

lr = LogisticRegression(random_state=0)

grid_search_lr = GridSearchCV(lr, param_grid_lr, cv=5)
grid_search_lr.fit(X_train_scaled, y_train)

best_params_lr = grid_search_lr.best_params_

# Rebuilding a model with best parameters
lr = LogisticRegression(C=best_params_lr['C'], random_state=0)
lr.fit(X_train_scaled, y_train)

test_score_train_lr = lr.score(X_train_scaled, y_train)
test_score_lr = lr.score(X_test_scaled, y_test)

# Output coeficients
coeficients_lr = lr.coef_
intercept_lr = lr.intercept_

# Predicting the Test set results
y_pred_lr = lr.predict(X_test_scaled)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm_lr = confusion_matrix(y_test, y_pred_lr)

# -----------------------------------------------------------------------------
# Classification by Random Forest
from sklearn.ensemble import RandomForestClassifier

param_grid_rf = {'n_estimators': [3, 10, 30, 50], 'max_features': [2, 20, 40]}

forest = RandomForestClassifier(random_state=0)

grid_search_rf = GridSearchCV(forest, param_grid_rf, cv=5)
grid_search_rf.fit(X_train_scaled, y_train)

best_params_rf = grid_search_rf.best_params_

# Rebuilding a model with best parameters
forest = RandomForestClassifier(
        n_estimators=best_params_rf['n_estimators'],
        max_features=best_params_rf['max_features'],
        random_state=0)
forest.fit(X_train_scaled, y_train)

test_score_train_rf = forest.score(X_train_scaled, y_train)
test_score_rf = forest.score(X_test_scaled, y_test)

# Output coeficients
feature_importances_rf = forest.feature_importances_

# Predicting the Test set results
y_pred_rf = forest.predict(X_test_scaled)

# Making the Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)

# -----------------------------------------------------------------------------
# Classification by Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(random_state=0)
gbc.fit(X_train_scaled, y_train)

test_score_train_gbc = gbc.score(X_train_scaled, y_train)
test_score_gbc = gbc.score(X_test_scaled, y_test)

# Output coeficients
feature_importances_gbc = gbc.feature_importances_

# Predicting the Test set results
y_pred_gbc = gbc.predict(X_test_scaled)

# Making the Confusion Matrix
cm_gbc = confusion_matrix(y_test, y_pred_gbc)