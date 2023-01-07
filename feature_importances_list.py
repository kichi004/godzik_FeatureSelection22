import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def avg_feature_importances(file_name, feature_names, trials = 100):
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

    for name, importance in zip(feature_names, mean_importances):
        print(f'{name}: {importance:.3f}')
