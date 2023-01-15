import accuracy_finder
import pandas as pd
import numpy as np
import statistics
from sklearn.ensemble import RandomForestClassifier

# sort function
def sorted_df(file_name, n = 500, depth_input = None):
    # load dataset
    df = pd.read_csv(file_name)

    # split dataset into features and targets
    feats = df.drop('Group', axis=1)
    labels = df['Group']

    # create empty list to store importances
    importances = []

    # repeat finding importances n times 
    for i in range(n):

        # create and fit classifier
        rfc = RandomForestClassifier(max_depth = depth_input)
        rfc.fit(feats, labels)

        # get the feature importances and store in list
        importances.append(rfc.feature_importances_)

    # calculate the average feature importances for each feature
    avg_importances = sum(importances) / len(importances)

    # create list of tuples and sort by feature importances
    feature_importances = [(feature, importance) for feature, importance in zip(feats.columns, avg_importances)]
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

    # Get the sorted column names
    sorted_columns = [tuple[0] for tuple in feature_importances]
    
    # Sort the dataframe by the sorted column names
    sorted_df = feats[sorted_columns]

    sorted_df.insert(0, labels.name, labels)

    sorted_df.to_csv("_sorted_features.csv", index = False)

# print importances list + get top n


# search through top n's function
def best_n_features_search(file_name, trials_input = 20, depth_input = None):
    # call sort function
    sorted_df(file_name, 500, depth_input)

    # load data
    full_sorted_df = pd.read_csv("_sorted_features.csv", index_col = False)
    width = len(full_sorted_df.columns)

    for n in range(2, width + 2):
        # select the first n columns
        first_n = full_sorted_df.iloc[:,:n]

        feats = first_n.drop(['Group'], axis = 1) # everything but the first column
        labels = first_n['Group'] # just the first column

        accuracy_list = []

        for i in range(trials_input):
            accuracy_list.append(accuracy_finder.find_accuracy(feats, labels, depth_input))

        accuracy = statistics.mean(accuracy_list)
        std_dev = statistics.stdev(accuracy_list)
        print(f"Accuracy for {n-1} features: {accuracy} +- {std_dev}%")

