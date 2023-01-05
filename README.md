# godzik_COVIDFeatureSelection22
 
missing_imputer.py = uses SimpleImputer to impute missing data and create "_imputed_data.csv"
loocv_accuracy.py = find_accuracy('file') finds the accuracy of RFC based on leave-one-out cross validation
n_estimators_search.py = searches and finds the best value of "n_estimators" given potential values

ordered_feature_selector.py = select_features(n) finds the top 'n' features using the RFC features_importances_ function
ordered_top_feature_search.py = determines the optimal number of top 'n' features to use in RFC search

changed Male = 1 and Female = 0