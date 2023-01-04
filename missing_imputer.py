# used to fill in missing values of dataset, currently filling in with mean/avg

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# load data into a pandas dataframe
df = pd.read_csv("FatalVSSurvival2.csv")

# creates a simple imputer detecing missing values 'NaN' and replacing with the mean of the column
simp = SimpleImputer(missing_values = np.nan, strategy = 'mean')

# creates a column transformer object with the imputer
ctf = ColumnTransformer([('imp', simp, [9, 12, 13, 19, 23, 24])])

# imputes missing values 
imputed = ctf.fit_transform(df)

# outputs to a new csv file 'imputed.csv'
imputed_df = pd.DataFrame(imputed)
imputed_df.to_csv("imputed2.csv", index = False)