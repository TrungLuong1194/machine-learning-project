# Importing the libraries
import numpy as np
import pandas as pd

#------------------------------------------------------------------------------
# Utilities
#------------------------------------------------------------------------------

# Describe data
def describe_dataframe(df=pd.DataFrame()):
    """
    This function generates descriptive stats of a dataframe
    Args:
        df (dataframe): the dataframe to be analyzed
    Return:
        None
    """
    
    print("\n\n")
    print("*" * 30)
    print("About the Data")
    print("*" * 30)
    
    print("Number of rows:", df.shape[0])
    print("Number of columns:", df.shape[1])
    print("\n")
    
    print("Column Names:", df.columns.values)
    print("\n")
    
    print("Column Data Types:\n", df.dtypes)
    print("\n")
    
    print("Columns with Missing Values:", 
          df.columns[df.isnull().any()].tolist())
    print("\n")
    
    print("Number of rows with Missing Values:", np.count_nonzero(df.isnull()))
    print("\n")
          
    print("General Stats:")
    print(df.info())
    print("\n")
    
    print("Summary Stats:")
    print(df.describe())
    print("\n")
    
    print("Sample Rows:")
    print(df.head())
    
#------------------------------------------------------------------------------
# Data preprocessing
#------------------------------------------------------------------------------

# Importing the dataset
dataset = pd.read_csv('income_evaluation.csv')

# Moving from "?" to NAN
dataset = dataset.replace(' ?', np.nan)

# Describe data
describe_dataframe(dataset)

# Imputing missing data
dataset = dataset.fillna(" Unknown")

# Check data
describe_dataframe(dataset)

#------------------------------------------------------------------------------
# Data Summarization
#------------------------------------------------------------------------------

print("Native Country:\n")
print(dataset['native-country'].value_counts())
print("\n")

print("Workclass:\n")
print(dataset['workclass'].value_counts())
print("\n")

print("Education:\n")
print(dataset['education'].value_counts())
print("\n")

#------------------------------------------------------------------------------
# Data Visualizing
#------------------------------------------------------------------------------
import matplotlib.pyplot as plt

params = {'legend.fontsize': 'small',
          'figure.figsize': (20, 10),
         'axes.labelsize': 'medium',
         'axes.titlesize':'medium',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}

plt.rcParams.update(params)

# Income density by age
dataset[['age', 'income']].hist(by='income', sharex=True)
plt.show()

# Income density by hours-per-week
dataset[['hours-per-week', 'income']].hist(by='income', sharex=True)
plt.show()

# Workclass Distribution
labels_wc = dataset['workclass'].value_counts().index
sizes_wc = []
for i in dataset['workclass'].value_counts().values:
    sizes_wc.append(i * 100 / sum(dataset['workclass'].value_counts().values))

plt.pie(sizes_wc, labels=labels_wc, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Workclass Distribution")
plt.show()

# Occupation Distribution
labels_oc = dataset['occupation'].value_counts().index
sizes_oc = []
for i in dataset['occupation'].value_counts().values:
    sizes_oc.append(i * 100 / sum(dataset['occupation'].value_counts().values))

plt.pie(sizes_oc, labels=labels_oc, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Occupation Distribution")
plt.show()

# Education Distribution
labels_ed = dataset['education'].value_counts().index
sizes_ed = []
for i in dataset['education'].value_counts().values:
    sizes_ed.append(i * 100 / sum(dataset['education'].value_counts().values))

plt.pie(sizes_ed, labels=labels_ed, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Education Distribution")
plt.show()

# Relationship Distribution
labels_re = dataset['relationship'].value_counts().index
sizes_re = []
for i in dataset['relationship'].value_counts().values:
    sizes_re.append(i * 100 / sum(dataset['relationship'].value_counts().values))

plt.pie(sizes_re, labels=labels_re, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Relationship Distribution")
plt.show()

#------------------------------------------------------------------------------
# Feature Engineering
#------------------------------------------------------------------------------

# 'fnlwgt' Log Transform
dataset['fnlwgt-log'] = np.log((1 + dataset['fnlwgt']))

fig, ax = plt.subplots()
dataset['fnlwgt-log'].hist()
ax.set_title('fnlwgt-log Distribution', fontsize=12)
plt.show()

# 'capital-gain' Log Transform
dataset['capital-gain-log'] = np.log((1 + dataset['capital-gain']))

fig, ax = plt.subplots()
dataset['capital-gain-log'].hist()
ax.set_title('capital-gain-log Distribution', fontsize=12)
plt.show()

# 'capital-loss' Log Transform
dataset['capital-loss-log'] = np.log((1 + dataset['capital-loss']))

fig, ax = plt.subplots()
dataset['capital-loss-log'].hist()
ax.set_title('capital-loss-log Distribution', fontsize=12)
plt.show()

# Feature scaling
from sklearn.preprocessing import MinMaxScaler

# Scaler 'age'
scaler_age = MinMaxScaler()
dataset['age-scaler'] = scaler_age.fit_transform(dataset[['age']])

# Scaler 'education-num'
scaler_edu = MinMaxScaler()
dataset['education-num-scaler'] = scaler_edu.fit_transform(dataset[['education-num']])

# Scaler 'hours-per-week'
scaler_hpw = MinMaxScaler()
dataset['hours-per-week-scaler'] = scaler_hpw.fit_transform(dataset[['hours-per-week']])

# Scaler 'fnlwgt-log'
scaler_fnlwgt = MinMaxScaler()
dataset['fnlwgt-log-scaler'] = scaler_fnlwgt.fit_transform(dataset[['fnlwgt-log']])

# Scaler 'capital-gain-log'
scaler_capital_gain = MinMaxScaler()
dataset['capital-gain-log-scaler'] = scaler_capital_gain.fit_transform(
                                                dataset[['capital-gain-log']])

# Scaler 'capital-loss-log'
scaler_capital_loss = MinMaxScaler()
dataset['capital-loss-log-scaler'] = scaler_capital_loss.fit_transform(
                                                dataset[['capital-loss-log']])

# Remove Duplicate features
dataset = dataset.drop(['age', 'fnlwgt', 'education-num', 'capital-gain',
                        'capital-loss', 'hours-per-week', 'fnlwgt-log',
                        'capital-gain-log', 'capital-loss-log'], axis=1)

# Encoding target field (feature 'income')
dataset = dataset.replace(' >50K', 1)
dataset = dataset.replace(' <=50K', 0)

# Setting features, targets
target = dataset['income']
feature = dataset.drop('income', axis=1)

# Dummy Coding
feature = pd.get_dummies(feature)

#------------------------------------------------------------------------------
# Feature Selection
#------------------------------------------------------------------------------

print('Feature set data [shape: '+str(feature.shape)+']')
print('Feature names:')
print(np.array(feature.columns), '\n')

from sklearn.feature_selection import chi2, SelectKBest

skb = SelectKBest(score_func=chi2, k = 70)
skb.fit(feature, target)

feature_scores = [(item, score) for item, score in zip(
                                feature.columns, skb.scores_)]   
sorted(feature_scores, key=lambda x: -x[1])[:70]

select_features_kbest = skb.get_support()
features_names_kbest = feature.columns[select_features_kbest]
features_selection = feature[features_names_kbest]

#------------------------------------------------------------------------------
# Setting Training set vs Test set
#------------------------------------------------------------------------------

# Setting X, y
X = features_selection.values
y = target.values

# Splitting the dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=0)
#------------------------------------------------------------------------------
# Building Model
#------------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Classification by Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

param_grid_lr = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

lr = LogisticRegression(random_state=0)

grid_search_lr = GridSearchCV(lr, param_grid_lr, cv=5)
grid_search_lr.fit(X_train, y_train)

best_params_lr = grid_search_lr.best_params_

# Rebuilding a model with best parameters
lr = LogisticRegression(C=best_params_lr['C'], random_state=0)
lr.fit(X_train, y_train)

test_score_train_lr = lr.score(X_train, y_train)
test_score_lr = lr.score(X_test, y_test)

# Output coeficients
coeficients_lr = lr.coef_
intercept_lr = lr.intercept_

# Predicting the Test set results
y_pred_lr = lr.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm_lr = confusion_matrix(y_test, y_pred_lr)

# -----------------------------------------------------------------------------
# Classification by Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(random_state=0)
gbc.fit(X_train, y_train)

test_score_train_gbc = gbc.score(X_train, y_train)
test_score_gbc = gbc.score(X_test, y_test)

# Output coeficients
feature_importances_gbc = gbc.feature_importances_

# Predicting the Test set results
y_pred_gbc = gbc.predict(X_test)

# Making the Confusion Matrix
cm_gbc = confusion_matrix(y_test, y_pred_gbc)

# -----------------------------------------------------------------------------
# Classification by Random Forest
from sklearn.ensemble import RandomForestClassifier

param_grid_rf = {'n_estimators': [3, 10, 30, 50], 'max_features': [2, 20, 40]}

forest = RandomForestClassifier(random_state=0)

grid_search_rf = GridSearchCV(forest, param_grid_rf, cv=5)
grid_search_rf.fit(X_train, y_train)

best_params_rf = grid_search_rf.best_params_

# Rebuilding a model with best parameters
forest = RandomForestClassifier(
        n_estimators=best_params_rf['n_estimators'],
        max_features=best_params_rf['max_features'],
        random_state=0)
forest.fit(X_train, y_train)

test_score_train_rf = forest.score(X_train, y_train)
test_score_rf = forest.score(X_test, y_test)

# Output coeficients
feature_importances_rf = forest.feature_importances_

# Predicting the Test set results
y_pred_rf = forest.predict(X_test)

# Making the Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)

#------------------------------------------------------------------------------
# Algorithm Evaluation
#------------------------------------------------------------------------------

evaluation_lr = np.array([cm_lr[0][0], cm_lr[1][1]])
evaluation_gbc = np.array([cm_gbc[0][0], cm_gbc[1][1]])
evaluation_rf = np.array([cm_rf[0][0], cm_rf[1][1]])

data = {'Logistic Regression': evaluation_lr,
        'Gradient Boosting Classifier': evaluation_gbc,
        'Random Forest': evaluation_rf}

evalution_algorithm = pd.DataFrame(data, index=['TN', 'TP'])