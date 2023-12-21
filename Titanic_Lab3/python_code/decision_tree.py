import data as data
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

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
    for train_index, test_index in data.kf.split(data.X):
        Xi_train, Xi_test = data.X.iloc[train_index], data.X.iloc[test_index]
        yi_train, yi_test = data.y.iloc[train_index], data.y.iloc[test_index]

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
    print('Accuracy( criterion = ', criterion, ') =', '{:.2f} %'.format(round(np.mean(accuracies) * 100, 2)))
    print('Std( criterion =', criterion, ')\t\t =', '{:.4f}'.format(round(np.std(accuracies), 4)))

    # Plot the best decision tree
    plt.figure()
    plot_tree(best_tree, filled=True, class_names=data.class_order, feature_names=data.data.columns, rounded=True)
    plt.tight_layout()
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

    for train_index, test_index in data.kf.split(data.X):
        Xi_train, Xi_test = data.X.iloc[train_index], data.X.iloc[test_index]
        yi_train, yi_test = data.y.iloc[train_index], data.y.iloc[test_index]

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
plt.figure()
plot_tree(best_tree, filled=True, class_names=data.class_order, feature_names=data.data.columns, rounded=True)
plt.tight_layout()
plt.savefig('../images/decision_tree.png', dpi=800)
plt.show()
