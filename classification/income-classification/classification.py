# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('income_evaluation.csv')

# Categorizing variables
dataset_dummies = pd.get_dummies(dataset)

# Setting X, y
X = dataset_dummies.iloc[:, [i for i in range(108)]].values
y = dataset_dummies.iloc[:, 109].values

# Splitting the dataset
from sklearn.model_selection import train_test_split

## Split the dataset into train+validation set and test set
X_train_validation, X_test, y_train_validation, y_test = train_test_split(
        X, y, test_size=0.15, random_state=0)
## Split train+validation set into training and validation set
X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_validation, y_train_validation, test_size=0.15, random_state=0)

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train_validation)
X_train_validation_scaled = scaler.transform(X_train_validation)
X_train_scaled = scaler.transform(X_train)
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------------------------------
# Classification by Logistic Regression
# Fitting classifier to the Training set
from sklearn.linear_model import LogisticRegression

best_score_lr = 0

for C in [0.001, 0.01, 0.1, 1, 10, 100]:
    logreg = LogisticRegression(C=C, random_state=0)
    logreg.fit(X_train_scaled, y_train)
    # Evaluating the model on the validation set
    score = logreg.score(X_validation_scaled, y_validation)
    
    if score > best_score_lr:
        best_score_lr = score
        best_C_lr = C

# Rebuilding a model with best parameters
logreg = LogisticRegression(C=best_C_lr, random_state=0)
logreg.fit(X_train_validation_scaled, y_train_validation)

test_score_lr = logreg.score(X_test_scaled, y_test)

# Output coeficients
coeficients_lr = logreg.coef_
intercept_lr = logreg.intercept_

# Predicting the Test set results
y_pred_lr = logreg.predict(X_test_scaled)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)

# -----------------------------------------------------------------------------
# Classification by Random Forest
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=1000, max_features=40, random_state=0)
forest.fit(X_train_validation, y_train_validation)

test_score_rf = forest.score(X_test, y_test)

# Output coeficients
feature_importances_rf = forest.feature_importances_

# Predicting the Test set results
y_pred_rf = forest.predict(X_test)

# Making the Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)

# -----------------------------------------------------------------------------
# Classification by Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(
        max_depth=3, learning_rate=0.1, random_state=0)
gbc.fit(X_train_validation, y_train_validation)

test_score_gbc = gbc.score(X_test, y_test)

# Output coeficients
feature_importances_gbc = gbc.feature_importances_

# Predicting the Test set results
y_pred_gbc = gbc.predict(X_test)

# Making the Confusion Matrix
cm_gbc = confusion_matrix(y_test, y_pred_gbc)