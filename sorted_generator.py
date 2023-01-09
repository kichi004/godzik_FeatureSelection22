import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def sorted_df(data, target, n = 500):
    
    # create empty list to store importances
    importances = []

    # repeat finding importances n times 
    for i in range(n):

        # create and fit classifier
        rfc = RandomForestClassifier(n_estimators = 128)
        rfc.fit(data, target)

        # get the feature importances and store in list
        importances.append(rfc.feature_importances_)

    # calculate the average feature importances for each feature
    avg_importances = sum(importances) / len(importances)

    # create list of tuples and sort by feature importances
    feature_importances = [(feature, importance) for feature, importance in zip(data.columns, avg_importances)]
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

    # Get the sorted column names
    sorted_columns = [tuple[0] for tuple in feature_importances]
    
    # Sort the dataframe by the sorted column names
    sorted_df = data[sorted_columns]

    sorted_df.insert(0, target.name, target)

    sorted_df.to_csv("_sorted_features.csv", index = False)

# load dataset
df = pd.read_csv('_imputed_data.csv')

# split dataset into features and targets
dataset_features = df.drop('Group', axis=1)
dataset_targets = df['Group']

sorted_df(dataset_features, dataset_targets)