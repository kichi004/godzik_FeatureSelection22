# used to fill in missing values of dataset

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def impute_missing_values(file_name, strat = 'mean', imputed_columns = []):
    # load csv file into pandas dataframe
    df = pd.read_csv(file_name)

    # create simple imputer to replace value
    simp = SimpleImputer(missing_values = np.nan, strategy = strat)

    # fit imputation
    simp.fit(df[imputed_columns])

    # apply imputations
    df[imputed_columns] = simp.transform(df[imputed_columns])

    # print dataframe
    print(df.head())

    # output dataframe as csv file
    df.to_csv("_imputed_data.csv", index = False)
