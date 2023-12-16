import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load the dataset
df = pd.read_csv("cars.csv")

# Preprocess the dataframe
buying_order = ['low', 'med', 'high', 'vhigh']
maint_order = ['low', 'med', 'high', 'vhigh']
doors_order = ['2', '3', '4', '5more']
persons_order = ['2', '4', 'more']
lug_boot_order = ['small', 'med', 'big']
safety_order = ['low', 'med', 'high']
class_order = ['unacc', 'acc', 'good', 'vgood']

# Encoding the data frame
encoder = OrdinalEncoder(categories=[buying_order, maint_order, doors_order, persons_order, lug_boot_order, safety_order, class_order])
encoded_data = encoder.fit_transform(df)
df = pd.DataFrame(encoded_data, columns=df.columns)

# Split data frame in data and target
data = df.drop(columns=['class'])
target = df['class']

# Splitting the data into training and validation samples
X, X_validate, y, y_validate = train_test_split(data, target, test_size=0.2, random_state=42)

# Set up a 10-Fold cross-validation
kf = KFold(n_splits=10, random_state=42, shuffle=True)

if(0):
    '''
    Task 1 -- Classification with Naive Bayes Classifier
    '''

    # Create the Multinomial Naive Bayes model
    # TODO: ROC CURVE (ALSO FOR SVM, AVERAGE ALL 4 ONE VS REST CURVES INTO ONE AND COMPARE THOSE
    naive_bayes_classifier = MultinomialNB()

    # Initialize accuracies array
    accuracies = []

    # 10-fold cross-validation
    for train_index, test_index in kf.split(X):
        Xi_train, Xi_test = X.iloc[train_index], X.iloc[test_index]
        yi_train, yi_test = y.iloc[train_index], y.iloc[test_index]

        # Fit the model to the training data
        naive_bayes_classifier.fit(Xi_train, yi_train)

        # Predict probabilities for each class for the test data
        yi_pred = naive_bayes_classifier.predict(Xi_test)

        # Calculate accuracy and append to the accuracies list
        accuracy = accuracy_score(yi_test, yi_pred)
        accuracies.append(accuracy)

    # Calculate and print results: accuracy mean and standard deviation
    print('Naive Bayes Classifier:')
    print('Accuracy =', '{:.2f} %'.format(round(np.mean(accuracies)*100, 2)))
    print('Std\t\t =', '{:.4f}'.format(round(np.std(accuracies), 4)))


    '''
    Task 2 -- Classification with Nearest Neighbor Classifier
    
    -> Find best number of nearest neighbors using accuracy
    '''

    print('\n----------------------------------------')

    # Initialize accuracy array for plot
    accuracy_by_k = []

    # 10-fold cross-validation
    for k in range(1, 21):
        knn_classifier = KNeighborsClassifier(n_neighbors=k)

        # Initialize accuracy array for classifier
        accuracies = []

        for train_index, test_index in kf.split(X):
            Xi_train, Xi_test = X.iloc[train_index], X.iloc[test_index]
            yi_train, yi_test = y.iloc[train_index], y.iloc[test_index]

            # Fit the model to the training data
            knn_classifier.fit(Xi_train, yi_train)

            # Predict probabilities for each class for the test data
            yi_pred = knn_classifier.predict(Xi_test)

            # Calculate accuracy and append to the accuracies list
            accuracy = accuracy_score(yi_test, yi_pred)
            accuracies.append(accuracy)

        accuracy_by_k.append(np.mean(accuracies))

        print('\nNearest Neighbor Classifier k =', k, ':')
        print('Accuracy =', '{:.2f} %'.format(round(np.mean(accuracies)*100, 2)))
        print('Std\t\t =', '{:.4f}'.format(round(np.std(accuracies), 4)))

    plt.plot(list(range(1, 21)), accuracy_by_k)
    plt.title('Accuracy vs. Number of Neighbors (k)')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.xticks(list(range(1, 21)))
    plt.savefig('../images/nearest_neighbor_accuracy.png', dpi=300)
    plt.show()


    '''
    Task 3 -- Classification with Decision Tree Classifier
    '''

    print('\n----------------------------------------')

    # Add criterion values to test
    criterions = ['gini', 'entropy']

    # Initialize best tree
    best_tree = DecisionTreeClassifier()

    # Test the criterion values
    for criterion in criterions:
        # Create the Multinomial Naive Bayes model
        decision_tree_classifier = DecisionTreeClassifier(criterion=criterion, random_state=42)

        # Initialize accuracies array
        accuracies = []

        # 10-fold cross-validation
        for train_index, test_index in kf.split(X):
            Xi_train, Xi_test = X.iloc[train_index], X.iloc[test_index]
            yi_train, yi_test = y.iloc[train_index], y.iloc[test_index]

            # Fit the model to the training data
            decision_tree_classifier.fit(Xi_train, yi_train)

            # Predict probabilities for each class for the test data
            yi_pred = decision_tree_classifier.predict(Xi_test)

            # Calculate accuracy and append to the accuracies list
            accuracy = accuracy_score(yi_test, yi_pred)
            accuracies_before = list(accuracies)
            accuracies.append(accuracy)

            # Update best tree if accuracy is biggest
            if accuracies_before != [] and accuracy > max(accuracies_before):
                best_tree = decision_tree_classifier

        # Calculate and print results: accuracy mean and standard deviation
        print('Decision Tree Classifier:\n')
        print('Accuracy( criterion = ', criterion,') =', '{:.2f} %'.format(round(np.mean(accuracies)*100, 2)))
        print('Std( criterion =', criterion, ')\t\t =', '{:.4f}'.format(round(np.std(accuracies), 4)))

        # Plot the best decision tree
        plt.figure(figsize=(12, 8))
        plot_tree(best_tree, filled=True, class_names=class_order, feature_names=data.columns, rounded=True)
        plt.savefig('../images/decision_tree_entropy.png', dpi=800)
        plt.show()


    # Test the max_depth hyperparameter

    # Initialize accuracy array for plot
    accuracy_by_n = []

    # Initialize max_depths to test
    max_depths = range(1, 21)

    # Initialize best tree
    best_tree = DecisionTreeClassifier()

    # 10-fold cross-validation
    for n in max_depths:
        decision_tree_classifier = DecisionTreeClassifier(max_depth=n)

        # Initialize accuracy array for classifier
        accuracies = []

        for train_index, test_index in kf.split(X):
            Xi_train, Xi_test = X.iloc[train_index], X.iloc[test_index]
            yi_train, yi_test = y.iloc[train_index], y.iloc[test_index]

            # Fit the model to the training data
            decision_tree_classifier.fit(Xi_train, yi_train)

            # Predict probabilities for each class for the test data
            yi_pred = decision_tree_classifier.predict(Xi_test)

            # Calculate accuracy and append to the accuracies list
            accuracy = accuracy_score(yi_test, yi_pred)
            accuracies_before = list(accuracies)
            accuracies.append(accuracy)

            # Update best tree if accuracy is biggest
            if accuracies_before != [] and n == 10 and accuracy > max(accuracies_before):
                best_tree = decision_tree_classifier

        accuracy_by_n.append(np.mean(accuracies))

        print('\nDecision Tree Classifier max_depth =', n, ':')
        print('Accuracy =', '{:.2f} %'.format(round(np.mean(accuracies) * 100, 2)))
        print('Std\t\t =', '{:.4f}'.format(round(np.std(accuracies), 4)))

    # plot Accuracy vs. Maximum Depth
    plt.plot(list(max_depths), accuracy_by_n)
    plt.title('Accuracy vs. Maximum Depth')
    plt.xlabel('Maximum Depth')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.xticks(list(max_depths))
    plt.savefig('../images/decision_tree_max_depth_accuracy.png', dpi=300)
    plt.show()

    # Plot the best decision tree
    plt.figure(figsize=(12, 8))
    plot_tree(best_tree, filled=True, class_names=class_order, feature_names=data.columns, rounded=True)
    plt.savefig('../images/decision_tree.png', dpi=800)
    plt.show()


'''
Task 4 -- Classification with Support Vector Machine
'''



