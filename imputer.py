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

def impute_missing_values_split(file_name, strat = 'mean'):
    # load csv file into pandas dataframe
    df = pd.read_csv(file_name)

    # create simple imputer to replace value
    simp = SimpleImputer(missing_values = np.nan, strategy = strat)

    # split df
    df_class = df['Group'].copy()
    df.drop(['Group'], axis = 1, inplace = True)

    # save column names
    column_names = list(df.columns.values)

    # fit imputation
    simp.fit(df)

    # apply imputations
    array = simp.transform(df)
    df = pd.DataFrame(array, columns = column_names)

    # recombine
    df = pd.concat((df_class, df), axis = 1)

    # print dataframe
    print(df.head())

    # output dataframe as csv file
    df.to_csv("_imputed_data.csv", index = False)
