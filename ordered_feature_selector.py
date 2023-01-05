# takes imputed dataset, then trains a classifier to get the most important n features as a dataset

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def select_features(n):
    # load dataset
    df = pd.read_csv('_imputed_data.csv')

    # split dataset into features and targets
    dataset_features = df.drop('Group', axis=1)
    dataset_targets = df['Group']

    # train random forest classifier
    rfc = RandomForestClassifier()
    rfc.fit(dataset_features, dataset_targets)

    # get feature importances
    importances = rfc.feature_importances_

    # sort features by importance in descending order
    indices = np.argsort(importances)[::-1]

    # Select the top n important features
    top_n_features = [dataset_features.columns[i] for i in indices[:n]]

    # display the top n important features
    # print(top_n_features)

    # create new dataframe with the top features
    df_important = df[['Group'] + top_n_features]

    # output the new dataframe to a CSV file
    df_important.to_csv('_top_features.csv', index=False)
