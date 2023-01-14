import imputer

while True:
    choice_input = input("Enter the value for the desired task.\n2. Impute Missing\n")

    if choice_input == '1':
        # everything
        print('hello')
        break

    if choice_input == '2':

        # select file
        file_input = input("Enter the name of the CSV file: ")

        # select version number for columns imputed
        version_number = input("Enter the version number of the file: ")

        selected_columns = []
        if version_number == '1' or version_number == 'v1':
            selected_columns = ['Resistin', 'D-dimer']
        elif version_number == '2' or version_number == 'v2':
            selected_columns = ['Resistin']
        
        # adding spacing
        print()

        # call function / CHANGE IMPUTATION TYPE HERE
        imputer.impute_missing_values(file_input, 'mean', selected_columns)
        break

    else:
        print('An acceptable value was not entered.')
        continue

