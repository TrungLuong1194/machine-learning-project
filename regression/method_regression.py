# Dataset: Data of Cars

# ---------------------------------------------------------------------------------
# Importing the libraries
# ---------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------------
# Importing the dataset
# ---------------------------------------------------------------------------------
dataset = pd.read_csv('car-data.csv')
X = dataset.iloc[:, [0, 1, 3, 4, 5]].values
y = dataset.iloc[:, 2].values

# ---------------------------------------------------------------------------------
# Encoding categorical data
# ---------------------------------------------------------------------------------
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

le_X = LabelEncoder()

# Index column of car_name, year
index_car_name = len(set(X[:, 0]))
index_year = len(set(X[:, 1])) + index_car_name

# Encoding car_name, year, fuel_type
X[:, 0] = le_X.fit_transform(X[:, 0])
X[:, 1] = le_X.fit_transform(X[:, 1])
X[:, 4] = le_X.fit_transform(X[:, 4])

ct = ColumnTransformer(
		[('one_hot_encoder', OneHotEncoder(categories='auto'), [0, 1, 4])],
		remainder='passthrough'
	)
X = ct.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = np.delete(X, [0, index_car_name, index_year], axis=1)

# ---------------------------------------------------------------------------------
# Splitting the dataset into the Training set and Test set
# ---------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ---------------------------------------------------------------------------------
# Decision Tree Regression
# Fitting the Decision Tree Regression Model to the dataset
# ---------------------------------------------------------------------------------
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred_dtr = regressor.predict(X_test)

# ---------------------------------------------------------------------------------
# Random Forest Regression
# Fitting the Random Forest Regression to the dataset
# ---------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred_rfr = regressor.predict(X_test)






print('-------------------------------------------------------')
print('X:')
print('\n')
print(X)
print('\n')
print('\n')

print('-------------------------------------------------------')
print('y:')
print('\n')
print(y)
print('\n')
print('\n')

print('-------------------------------------------------------')
print('X train:')
print('\n')
print(X_train)
print('\n')
print('\n')

print('-------------------------------------------------------')
print('y train:')
print('\n')
print(y_train)
print('\n')
print('\n')

print('-------------------------------------------------------')
print('X test:')
print('\n')
print(X_test)
print('\n')
print('\n')

print('-------------------------------------------------------')
print('y test:')
print('\n')
print(y_test)
print('\n')
print('\n')

print('-------------------------------------------------------')
print('Decision Tree Regression\n')
print('y predict:')
print('\n')
print(y_pred_dtr)
print('\n')
print('\n')

print('-------------------------------------------------------')
print('Random Forest Regression\n')
print('y predict:')
print('\n')
print(y_pred_rfr)
print('\n')
print('\n')