import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz, DecisionTreeClassifier
from pydot import graph_from_dot_file

def visualize_tree_classifier(file_name, output_name, depth_input = None, splitting = 'gini'):
    # read in data
    df = pd.read_csv(file_name)

    # split dataset into label columns and feature columns
    feats = df.drop(['Group'], axis = 1) # everything but the first column
    labels = df['Group'] # just the first column

    # generate classifier
    rfc = DecisionTreeClassifier(max_depth = depth_input, criterion = splitting)

    # train classifier
    rfc.fit(feats, labels)

    # extract single tree
    # estimator = rfc.estimators_[5]

    # get feature_names
    feature_names = list(df.columns.values)
    feature_names.pop(0)
    label_names = ['Fatality', 'Survival']

    # export as dot file
    export_graphviz(rfc, out_file = 'tree.dot', feature_names = feature_names, class_names = label_names, impurity = False, rounded = True, proportion = False, precision = 2, filled = True)

    # convert to png file
    (graph,) = graph_from_dot_file('tree.dot')
    graph.write_png(output_name)

def visualize_forest_classifier(file_name, output_base, depth_input = None):
    # read in data
    df = pd.read_csv(file_name)

    # split dataset into label columns and feature columns
    feats = df.drop(['Group'], axis = 1) # everything but the first column
    labels = df['Group'] # just the first column

    # generate classifier
    rfc = RandomForestClassifier(n_estimators = 100, max_depth = depth_input)

    # train classifier
    rfc.fit(feats, labels)

    for i in range(20):
        # extract single tree
        estimator = rfc.estimators_[i]

        # get feature_names
        feature_names = list(df.columns.values)
        feature_names.pop(0)
        label_names = ['Survival', 'Fatality']

        # export as dot file
        export_graphviz(estimator, out_file = 'tree.dot', feature_names = feature_names, class_names = label_names, impurity = False, rounded = True, proportion = False, precision = 2, filled = True)

        # make output name 
        number_string = str(i) 
        output_name = output_base + '_' + number_string + '.png'

        # convert to png file
        (graph,) = graph_from_dot_file('tree.dot')
        graph.write_png(output_name)


