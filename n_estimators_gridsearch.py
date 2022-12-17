# used to determine the best value for n_estimators in random forest classifier

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# load dataset into pandas dataframe
df = pd.read_csv("imputed.csv")

# split dataset into label columns and feature columns
feats = df.drop(['0'], axis = 1) # everything but the first column
labels = df['0'] # just the first column

# define parameters for possible n_estimator values
# trees_parameters = {'n_estimators': [10, 50, 100, 150, 200, 250, 500]} # returned 50, 10, 50
# trees_parameters = {'n_estimators': [10, 20, 30, 40, 50, 60, 70]} # returned 10, 30, 50, 50
trees_parameters = {'n_estimators': [10, 20, 30, 35, 40, 45, 50, 55]} # returned 20, 45, 20, 35, 40, 35

# initialize random forest classifier model
rfc_model = RandomForestClassifier()

# initialize grid search object
grid_search = GridSearchCV(rfc_model, trees_parameters, cv = 10, scoring = 'accuracy')

# fit grid search to training data
grid_search.fit(feats, labels)
print(f'Best value for n_estimators: {grid_search.best_params_["n_estimators"]}')

# currently using 35 n_estimators based on GridSearchCV









