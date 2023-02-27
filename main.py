import imputer
import accuracy_finder
import greedy_feature_selector
import importance_feature_selector
import visualizer
import pandas as pd
import time

while True:
    choice_input = input("\nEnter the value for the desired task.\n1. Everything | 2. Impute | 3. Accuracy | 4. Greedy Search | 5. Importances Search | \n6. Get Importance Values | 7. Get Top N Features | 8. Visualize Tree | 9. Visualize Forest | 10. Custom | 11. Get Scores \n")
    tic = time.perf_counter()

    if choice_input == '1': # everything >> imputes, greedy search and importance search
        # select file
        file_input = input("Enter the name of the CSV file: ")

        tree_visual_name = "greedy_tree_visual_"
        forest_visual_name = "nonselected_tree_visual_"
        top_feature_visual_name = "top_feature_tree_visual_"

        # select version number for columns imputed
        version_number = input("Enter the version number of the file: ")
        selected_columns = []
        if version_number == '0' or version_number == 'v0':
            tree_visual_name = tree_visual_name + "v0.png"
            forest_visual_name = forest_visual_name + "v0.png"
            top_feature_visual_name = top_feature_visual_name + "v0.png"
        elif version_number == '1' or version_number == 'v1':
            tree_visual_name = tree_visual_name + "v1.png"
            forest_visual_name = forest_visual_name + "v1.png"
            top_feature_visual_name = top_feature_visual_name + "v1.png"
        elif version_number == '2' or version_number == 'v2':
            tree_visual_name = tree_visual_name + "v2.png"
            forest_visual_name = forest_visual_name + "v2.png"
            top_feature_visual_name = top_feature_visual_name + "v2.png"
        elif version_number == '3' or version_number == 'v3':
            tree_visual_name = tree_visual_name + "v3.png"
            forest_visual_name = forest_visual_name + "v3.png"
            top_feature_visual_name = top_feature_visual_name + "v3.png"
        elif version_number == 'c' or version_number == 'clinical':
            tree_visual_name = tree_visual_name + "clinical.png"
            forest_visual_name = forest_visual_name + "clinical.png"
            top_feature_visual_name = top_feature_visual_name + "clinical.png"
        elif version_number == 'm' or version_number == 'molecular':
            tree_visual_name = tree_visual_name + "molecular.png"
            forest_visual_name = forest_visual_name + "molecular.png"
            top_feature_visual_name = top_feature_visual_name + "molecular.png"
        elif version_number == 'clinic':
            tree_visual_name = tree_visual_name + "clinic.png"
            forest_visual_name = forest_visual_name + "clinic.png"
            top_feature_visual_name = top_feature_visual_name + "clinic.png"
        elif version_number == 'v4':
            tree_visual_name = tree_visual_name + "v4.png"
            forest_visual_name = forest_visual_name + "v4.png"
            top_feature_visual_name = top_feature_visual_name + "v4.png"
        elif version_number == 'c2':
            tree_visual_name = tree_visual_name + "clinical2.png"
            forest_visual_name = forest_visual_name + "clinical2.png"
            top_feature_visual_name = top_feature_visual_name + "clinical2.png"
        elif version_number == 'm2':
            tree_visual_name = tree_visual_name + "molecular2.png"
            forest_visual_name = forest_visual_name + "molecular2.png"
            top_feature_visual_name = top_feature_visual_name + "molecular2.png"
        
        # adding spacing
        print()

        # Index(['Diabetes', 'LCN2', 'OPN', 'AST'], dtype='object')  89.310%
        # Index(['Diabetes', 'LCN2', 'OPN', 'AST', 'Hispanic'], dtype='object')  90.345%

        # calls every function
        imputer.impute_missing_values_split(file_input, 'mean')
        print()
        highest_n = importance_feature_selector.best_n_features_search("_imputed_data.csv")
        print()
        importance_feature_selector.avg_feature_importances("_imputed_data.csv")
        print()
        acc = importance_feature_selector.get_top_n_features("_sorted_by_importances.csv", highest_n-1)
        print()
        greedy_feature_selector.greedy_fw_search("_sorted_by_importances.csv", acc)
        print()
        visualizer.visualize_tree_classifier("_greedy_select_result.csv", tree_visual_name)
        visualizer.visualize_tree_classifier("_imputed_data.csv", forest_visual_name)
        print()
        accuracy_finder.find_scores("_top_X_features.csv")
        accuracy_finder.find_scores("_greedy_select_result.csv")
        accuracy_finder.find_scores("_imputed_data.csv")
        break

    elif choice_input == '2': # imputes file
        # select file
        file_input = input("Enter the name of the CSV file: ")

        # select version number for columns imputed
        version_number = input("Enter the version number of the file: ")

        selected_columns = []
        if version_number == '0' or version_number == 'v0':
            selected_columns = ['Resistin', 'IL-6', 'IFNλ2/3', 'OPN', 'Cystatin C', 'D-dimer']
        elif version_number == '1' or version_number == 'v1':
            selected_columns = ['Resistin', 'OPN', 'D-dimer']
        elif version_number == '2' or version_number == 'v2':
            selected_columns = ['Resistin', 'OPN']
        elif version_number == '3' or version_number == 'v3':
            selected_columns = ['Resistin']
        elif version_number == 'c' or version_number == 'clinical':
            selected_columns = ['D-dimer']
        elif version_number == 'm' or version_number == 'molecular':
            selected_columns = ['Resistin', 'OPN']
        elif version_number == 'clinic':
            selected_columns = ['SBP', 'DBP', 'AST', 'ALT', 'LDH', 'CRP', 'D-dimer']
        
        # adding spacing
        print()

        # call function / CHANGE IMPUTATION TYPE HERE
        imputer.impute_missing_values(file_input, 'mean', selected_columns)

        # print success 
        print(f"\nMissing values of {file_input} were successfully imputed and saved as \'_imputed_data.csv\'\n")

        break

    elif choice_input == '3': # find accuracy
        # select file
        file_input = input("Enter the name of the CSV file: ")

        # select number of trials
        trial_input = input("Enter the number of trials to conduct (rec 20): ")
        trial_input = int(trial_input)

        # select max depth
        depth_input = input("Enter the maximum depth for the classifier (enter '0' for no maximum): ")
        depth_input = int(depth_input)
        if depth_input == 0:
            depth_input = None

        # add spacing
        print()

        # call function
        accuracy_finder.find_accuracy_trials(file_input, trial_input, depth_input)
        break

    elif choice_input == '4': # greedy forward selection
        # select file
        file_input = input("Enter the name of the CSV file: ")

        # select printing threshold
        threshold_input = input("Print any combinations of accuracy higher than what value?: ")
        threshold_input = int(threshold_input)
        
        # select max depth
        depth_input = input("Enter the maximum depth for the classifier (enter '0' for no maximum): ")
        depth_input = int(depth_input)
        if depth_input == 0:
            depth_input = None

        # add spacing
        print()

        # call function
        greedy_feature_selector.greedy_fw_search(file_input, threshold_input, depth_input)
        break

    elif choice_input == '5': # search by feature importance
        # select file
        file_input = input("Enter the name of the CSV file: ")

        # select number of trials
        trial_input = input("Enter the number of trials to conduct (rec = 20): ")
        trial_input = int(trial_input)

        # select max depth
        depth_input = input("Enter the maximum depth for the classifier (enter '0' for no maximum): ")
        depth_input = int(depth_input)
        if depth_input == 0:
            depth_input = None
        
        skip_input = input("Enter '1' if you would like to skip the initial sorting process: ")
        if skip_input == '1':
            skip_input = True
        else:
            skip_input = False

        # add spacing
        print()

        # call function
        importance_feature_selector.best_n_features_search(file_input, trial_input, depth_input, skip_input)
        break

    elif choice_input == '6': # calculate and list importance values of the features
        # select file
        file_input = input("Enter the name of the CSV file: ")

        # select number of trials
        trial_input = input("Enter the number of trials to conduct (rec = 100): ")
        trial_input = int(trial_input)

        # add spacing
        print()

        # call function
        importance_feature_selector.avg_feature_importances(file_input, trial_input)
        break

    elif choice_input == '7': # get the top n features
        # select file
        file_input = input("Enter the name of the CSV file: ")

        # select number of features
        feature_count_input = input("Enter the number of top features you would like to retrieve: ")
        feature_count_input = int(feature_count_input)

        # add spacing
        print()

        # call function
        importance_feature_selector.get_top_n_features(file_input, feature_count_input)
        break

    elif choice_input == '8': # get decision tree visualization
        # select file
        file_input = input("Enter the name of the CSV file: ")

        # select output file name
        output_name = input("Enter the name of the output file: ")

        # select max depth
        depth_input = input("Enter the maximum depth for the classifier (enter '0' for no maximum): ")
        depth_input = int(depth_input)
        if depth_input == 0:
            depth_input = None

        # add spacing
        print()

        # call function
        visualizer.visualize_tree_classifier(file_input, output_name, depth_input)
        break

    elif choice_input == '9': # get forest visualization
        # select file
        file_input = input("Enter the name of the CSV file: ")

        # select output file name
        output_name = input("Enter the name of the output file: ")

        # select max depth
        depth_input = input("Enter the maximum depth for the classifier (enter '0' for no maximum): ")
        depth_input = int(depth_input)
        if depth_input == 0:
            depth_input = None

        # add spacing
        print()

        # call function
        visualizer.visualize_forest_classifier(file_input, output_name, depth_input)
        break

    elif choice_input == '10': # custom dataset accuracy finder
        # select max depth
        depth_input = input("Enter the maximum depth for the classifier (enter '0' for no maximum): ")
        depth_input = int(depth_input)
        if depth_input == 0:
            depth_input = None
        
        # read data in
        df = pd.read_csv("_imputed_data.csv")

        # select custom set of features / MODIFY HERE
        custom_df = df[['Group','Diabetes', 'LCN2', 'OPN']]

        # convert to csv file
        custom_df.to_csv("custom.csv", index = False)

        # call trials accuracy finder
        print(f'Columns are: {custom_df.columns.values}')
        accuracy_finder.find_accuracy_trials("custom.csv", 20, depth_input)
        break

    elif choice_input == '11': # f1 score, precision, accuracy
        # select file
        file_input = input("Enter the name of the CSV file: ")

        # call function
        accuracy_finder.find_scores(file_input)
        break

    else:
        print('An acceptable value was not entered.')
        continue

toc = time.perf_counter()
print(f"Time elapsed: {toc - tic:.1f} seconds.")

