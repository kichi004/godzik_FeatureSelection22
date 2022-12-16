import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# load data into a pandas dataframe
df = pd.read_csv("FatalVSSurvival.csv", header = None)

# creates a simple imputer object detecing missing values 'NaN' 
simp = SimpleImputer(missing_values = np.nan, strategy = 'median')

# creates a column transformer object
ctf = ColumnTransformer([('imp', simp, [9, 12, 13, 24])])

# imputes missing values 
imputed = ctf.fit_transform(df)

# applies the changes to the original dataframe
print(imputed)
df.loc[:, 9] = imputed[:, 0]
df.loc[:, 12] = imputed[:, 1]
df.loc[:, 13] = imputed[:, 2]
df.loc[:, 24] = imputed[:, 3]

# outputs to a new csv file 'imputed.csv' without row names
df.to_csv("imputed.csv", index = False)