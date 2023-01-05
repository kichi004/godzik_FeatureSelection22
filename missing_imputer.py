# used to fill in missing values of dataset, currently filling in with mean/avg

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# load data into a pandas dataframe
df = pd.read_csv("_survival_data_initial.csv")

# creates a simple imputer detecing missing values 'NaN' and replacing with the mean of the column
simp = SimpleImputer(missing_values = np.nan, strategy = 'mean')

# create a list of the columns planned to be imputed 
imputed_cols = ['Resistin', 'IL-6', 'IFNλ2/3', 'OPN','Cystatin C', 'D-dimer']

# identify the imputed data based on the imputed columns
data_to_impute = df[imputed_cols]

# impute data
imputed_data = simp.fit_transform(data_to_impute)

# assigned imputed data to the original df
df[imputed_cols] = imputed_data

# outputs dataframe as csv file
df.to_csv("_imputed_data.csv", index = False)
