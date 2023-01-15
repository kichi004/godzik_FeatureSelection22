import loocv_accuracy
import pandas as pd
import numpy as np
import statistics

# load data
full_sorted_df = pd.read_csv("_sorted_features.csv", index_col = False)

for n in range(2, 26):
    # select the first n columns
    first_n = full_sorted_df.iloc[:,:n]

    # save the first n columns as csv
    first_n.to_csv("first_n_columns.csv", index = False)

    accuracy_list = []

    for i in range(20):
        accuracy_list.append(loocv_accuracy.find_accuracy("first_n_columns.csv"))

    accuracy = statistics.mean(accuracy_list)
    std_dev = statistics.stdev(accuracy_list)
    print(f"{n-1} top feature accuracy was {accuracy}% + {std_dev}")

