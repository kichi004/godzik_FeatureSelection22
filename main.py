import imputer
import accuracy_finder

while True:
    choice_input = input("\nEnter the value for the desired task.\n2. Impute | 3. Accuracy | \n")

    if choice_input == '1': # everything >> imputes, greedy search and importance search
        print('hello')
        break

    if choice_input == '2': # imputes file

        # select file
        file_input = input("Enter the name of the CSV file: ")

        # select version number for columns imputed
        version_number = input("Enter the version number of the file: ")

        selected_columns = []
        if version_number == '1' or version_number == 'v1':
            selected_columns = ['Resistin', 'OPN', 'D-dimer']
        elif version_number == '2' or version_number == 'v2':
            selected_columns = ['Resistin']
        
        # adding spacing
        print()

        # call function / CHANGE IMPUTATION TYPE HERE
        imputer.impute_missing_values(file_input, 'mean', selected_columns)

        # print success 
        print(f"\nMissing values of {file_input} were successfully imputed and saved as \'_imputed_data.csv\'\n")

        break

    if choice_input == '3': # find accuracy

        # select file
        file_input = input("Enter the name of the CSV file: ")

        # select number of trials
        trial_input = input("Enter the number of trials to conduct: ")
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

    else:
        print('An acceptable value was not entered.')
        continue

