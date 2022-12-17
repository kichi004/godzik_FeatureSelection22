# used to fill in missing values of dataset, currently filling in with mean/avg

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# load data into a pandas dataframe
df = pd.read_csv("FatalVSSurvival.csv", header = None)

# creates a simple imputer detecing missing values 'NaN' and replacing with the mean of the column
simp = SimpleImputer(missing_values = np.nan, strategy = 'mean')

# creates a column transformer object with the imputer
ctf = ColumnTransformer([('imp', simp, [9, 12, 13, 19, 23, 24])])

# imputes missing values 
imputed = ctf.fit_transform(df)

# applies the changes to the original dataframe
df.loc[:, 9] = imputed[:, 0]
df.loc[:, 12] = imputed[:, 1]
df.loc[:, 13] = imputed[:, 2]
df.loc[:, 19] = imputed[:, 3]
df.loc[:, 23] = imputed[:, 4]
df.loc[:, 24] = imputed[:, 5]

# outputs to a new csv file 'imputed.csv'
df.to_csv("imputed.csv", index = False)