import loocv_accuracy
import pandas as pd
import numpy as np

highest_overall_accuracy = 0
best_overall_features = pd.DataFrame()

# load data
df = pd.read_csv("_imputed_data.csv", index_col = False)

# split dataset into label columns and feature columns
features = df.drop(['Group'], axis = 1) # everything but the first column
labs = df[df.columns[0]] # just the first column

# create a list of available features
available = features
selected = pd.DataFrame()

# start loop to n = 24 features
for i in range(24):

    # generate best iteration accuracy and features storage
    best_iteration_features = pd.DataFrame()
    highest_iteration_accuracy = 0
    print(len(available.columns))

    # secondary loop, adding every individual features
    for n in range(len(available.columns)):

        # add previously selected features and add one features from the available
        current_df = pd.DataFrame()
        current_df = pd.concat([selected, available.iloc[:, n]], axis=1)

        # output and get accuracy of current iteration
        complete_df = pd.concat([labs, current_df], axis = 1)
        complete_df.to_csv("_current_iteration.csv", index = False)
        iteration_accuracy = loocv_accuracy.find_accuracy("_current_iteration.csv")
        column_name = available.columns[n]
        print(f"{n+1}. {column_name}: {iteration_accuracy}%")
        #if iteration_accuracy > 85:
        #    print(complete_df.columns)
        #    print(iteration_accuracy)

        # update highest_iteration accuracy and current_iteration list
        if iteration_accuracy >= highest_iteration_accuracy:
            best_iteration_features = current_df
            highest_iteration_accuracy = iteration_accuracy
    
    # compare to highest_overall_accuracy and update
    if highest_iteration_accuracy >= highest_overall_accuracy:
        highest_overall_accuracy = highest_iteration_accuracy
        best_overall_features = best_iteration_features
        #print(f'Current best is {highest_overall_accuracy}%')

    selected = best_iteration_features
    last_column = selected.columns[-1]
    available = available.drop(labels = last_column, axis = 1)
    available.to_csv("_available.csv", index = False)

print("best was ")
print(best_overall_features)
print(highest_overall_accuracy)
final = pd.concat([labs, best_overall_features], axis = 1)
final.to_csv("_greedy_final.csv", index = False)


    

    

