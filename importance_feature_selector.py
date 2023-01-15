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

    sorted_df.to_csv("_sorted_by_importances.csv", index = False)

# print importances list + get top n
def avg_feature_importances(file_name, trials = 100):
    # initialize empty list to store importances
    importances_list = []

    # load dataset
    df = pd.read_csv(file_name)

    # split dataset into label columns and feature columns
    feats = df.drop(['Group'], axis = 1) # everything but the first column
    labels = df['Group'] # just the first column

    # loop to run multiple trials
    for i in range(trials):

        # Create a random forest classifier
        rfc = RandomForestClassifier()

        # fit classifier to the data
        rfc.fit(feats, labels)

        # get feature importances
        importances = rfc.feature_importances_

        # append importances to the list
        importances_list.append(importances)

    # convert list to numpy array
    importances_array = np.array(importances_list)

    # Compute mean of importances
    mean_importances = np.mean(importances_array, axis=0)

    # get names of every feature
    feature_names = list(df.columns.values)
    feature_names.pop(0)

    for name, importance in zip(feature_names, mean_importances):
        print(f'{name}: {importance:.3f}')
    print()

# search through top n's function
def best_n_features_search(file_name, trials_input = 20, depth_input = None, skip = False):
    # call sort function
    if (not skip):
        sorted_df(file_name, 500, depth_input)

    # load data
    full_sorted_df = pd.read_csv("_sorted_by_importances.csv", index_col = False)
    width = len(full_sorted_df.columns)
    highest_n = 2
    highest_n_accuracy = 0

    for n in range(2, width + 2):
        # select the first n columns
        first_n = full_sorted_df.iloc[:,:n]

        feats = first_n.drop(['Group'], axis = 1) # everything but the first column
        labels = first_n['Group'] # just the first column

        accuracy_list = []

        for i in range(trials_input):
            accuracy_list.append(accuracy_finder.find_accuracy_split(feats, labels, depth_input))

        accuracy = statistics.mean(accuracy_list)
        std_dev = statistics.stdev(accuracy_list)

        print(f"Accuracy for {n-1} features: {accuracy:3f} +- {std_dev:3f}%")

        if accuracy > highest_n_accuracy:
            highest_n = n
            highest_n_accuracy = accuracy
    
    first_n_best = full_sorted_df.iloc[:,:highest_n]
    print(f"\nThe highest accuracy set had {highest_n-1} features with {highest_n_accuracy:3f}% accuracy.\n")
    print(first_n_best.head())
    return highest_n

def get_top_n_features(file_name, n):
    # read in data
    full_sorted_df = pd.read_csv(file_name, index_col = False)

    # retrieve a dataframe of the first n+1 features
    first_n = full_sorted_df.iloc[:,:n+1]

    # output as csv
    first_n.to_csv("_top_X_features.csv", index = False)
    print(first_n.head())
    print(f"\nThe top {n} features were outputted as \'_top_X_features.csv\'")

    # find accuracy
    accuracy_finder.find_accuracy_trials("_top_X_features.csv")




