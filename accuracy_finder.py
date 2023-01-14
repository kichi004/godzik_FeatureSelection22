# uses leave-one-out random forest classifier on an inputted dataset

import pandas as pd
import statistics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut

def find_accuracy(file_name, depth_input = None):
    # load dataset into pandas dataframe
    df = pd.read_csv(file_name)

    # split dataset into label columns and feature columns
    feats = df.drop(['Group'], axis = 1) # everything but the first column
    labels = df['Group'] # just the first column

    # Initialize leave-one-out object
    loo = LeaveOneOut()

    # Initialize random forest classifier 
    rfc_model = RandomForestClassifier(max_depth = depth_input)

    # Initialize a list to store the scores
    scores_list = []

    # Iterate through training and test sets generated by leave-one-out
    for train_index, test_index in loo.split(feats):
        # Split dataset into a training and test set
        feats_train, feats_test = feats.iloc[train_index], feats.iloc[test_index]
        labels_train, labels_test = labels.iloc[train_index], labels.iloc[test_index]

        # Fit model to current training set
        rfc_model.fit(feats_train, labels_train)

        # Predict labels for test set
        y_pred = rfc_model.predict(feats_test)

        # Calculate the score for this fold
        accuracy_score = rfc_model.score(feats_test, labels_test)
        scores_list.append(accuracy_score)

    # calculate and print the average of the accuracies generated by the cv
    # print(f'{statistics.mean(scores_list)*100:.2f} ({statistics.stdev(scores_list)*100:.2f})% for {scores_list.__len__()} samples')
    return statistics.mean(scores_list)*100

def find_accuracy_trials(file_name, trials = 20, depth_input = None):
    # list to store accuracy scores
    accuracy_list = []

    # run multiple trials
    for i in range(trials):
        accuracy_list.append(find_accuracy(file_name, depth_input))
    
    # calculate
    accuracy = statistics.mean(accuracy_list)
    std_dev = statistics.stdev(accuracy_list)

    print(f"Accuracy of {file_name} was {accuracy} +- {std_dev}%\n")











