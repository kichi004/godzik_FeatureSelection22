import accuracy_finder
import pandas as pd
import numpy as np

def greedy_fw_search(file_name, print_threshold, max_depth):
    highest_overall_accuracy = 0
    best_overall_features = pd.DataFrame()

    # load data
    df = pd.read_csv(file_name, index_col = False)

    # split dataset into label columns and feature columns
    features = df.drop(['Group'], axis = 1) # everything but the first column
    labs = df[df.columns[0]] # just the first column

    # create a list of available features
    available = features
    selected = pd.DataFrame()
    avail_columns = len(available.columns)

    # start loop to n = 24 features
    for i in range(avail_columns):

        # generate best iteration accuracy and features storage
        best_iteration_features = pd.DataFrame()
        highest_iteration_accuracy = 0

        # secondary loop, adding every individual features
        for n in range(len(available.columns)):

            # add previously selected features and add one features from the available
            current_df = pd.DataFrame()
            current_df = pd.concat([selected, available.iloc[:, n]], axis=1)

            # output and get accuracy of current iteration
            iteration_accuracy = accuracy_finder.find_accuracy_split(current_df, labs, max_depth)
            column_name = available.columns[n]
            
            # prints whatever feature was added and the iteration accuracy
            # print(f"{n+1}. {column_name}: {iteration_accuracy}%")

            # print if the iteraction_accuracy is higher than the threshold
            if iteration_accuracy > print_threshold:
                complete_df = pd.concat([labs, current_df], axis = 1)
                print(complete_df.columns, end = "")
                print(f" {iteration_accuracy: .2f}%")

            # update highest_iteration accuracy and current_iteration list
            if iteration_accuracy >= highest_iteration_accuracy:
                best_iteration_features = current_df
                highest_iteration_accuracy = iteration_accuracy
        
        # compare to highest_overall_accuracy and update
        if highest_iteration_accuracy >= highest_overall_accuracy:
            highest_overall_accuracy = highest_iteration_accuracy
            best_overall_features = best_iteration_features

        # removes the best feature of the iteration from the available dataframe
        selected = best_iteration_features
        last_column = selected.columns[-1]
        available = available.drop(labels = last_column, axis = 1)

    # final output 
    print("Highest Accuracy Dataset Found:\n")
    final = pd.concat([labs, best_overall_features], axis = 1)
    final.to_csv("_greedy_search_result.csv", index = False)
    print(final.head())
    accuracy_finder.find_accuracy_trials("_greedy_search_result.csv", 20, max_depth)
    