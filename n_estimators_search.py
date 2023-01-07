# used to determine the best value for n_estimators in random forest classifier

import pandas as pd
import statistics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# load dataset into pandas dataframe
df = pd.read_csv("_greedy_final1.csv")

# split dataset into label columns and feature columns
feats = df.drop(['Group'], axis = 1) # everything but the first column
labels = df['Group'] # just the first column

# define parameters for possible n_estimator values
trees_parameters = {'n_estimators': [64, 80, 96, 112, 128]}
best_params_list = []

# loops the following 
for i in range(100):
    # initialize random forest classifier model
    rfc_model = RandomForestClassifier()
    # initialize grid search object
    grid_search = GridSearchCV(rfc_model, trees_parameters, cv = 10, scoring = 'accuracy')

    # fit grid search to training data
    grid_search.fit(feats, labels)
    best_params_list.append(grid_search.best_params_["n_estimators"])

# calculate the average of the best_params_list
print(f'Average Best n_estimator value: {statistics.mean(best_params_list):.1f} estimators')
print(best_params_list)

# currently using 30 estimators based on results








