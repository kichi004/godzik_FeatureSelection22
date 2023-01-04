import statistics
import loocv_rf_classifier
import featureSelector

highest_accuracy_n = 1
highest_accuracy_percentage = 0

# loop going through number of features
for n in range (1, 25):

    # list of accuracy results for the n number of top features
    n_features_accuracies = []

    # arbitrary number of loops to get an average value
    for k in range(100):

        # run featureSelector for "n" to get dataset with n top features
        featureSelector.featureSelection(n)

        # get accuracy of the n-top-feature dataset classification and add to list
        n_features_accuracies.append(loocv_rf_classifier.loocv_rf_classification("important_features.csv"))

    # finds the average accuracy of the 30 trials for n top features selected
    average_accuracy = statistics.mean(n_features_accuracies)

    # print accuracy of the trial into the terminal
    print(f'Average accuracy of {n} top features trials was {average_accuracy:1f}%')

    if average_accuracy > highest_accuracy_percentage:
        highest_accuracy_n = n
        highest_accuracy_percentage = average_accuracy

print(f'{highest_accuracy_n} number of features had the highest accuracy, with a value of {highest_accuracy_percentage} %.')


