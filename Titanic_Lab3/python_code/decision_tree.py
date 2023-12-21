import pandas as pd

import data as data
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, plot_tree


"""
Usando el clasificador DecisionTree, ajustando los hiperparametros 'criterion' y 'max_depth',
se puede obtener un accuracy de 77.11% con el criterion 'entropy' y un 'max_depth' de 3.
El AUC es 0.9875.
A continuación vemos el entrenamiento.
"""


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
    for train_index, test_index in data.kf.split(data.X_train):
        Xi_train, Xi_test = data.X_train.iloc[train_index], data.X_train.iloc[test_index]
        yi_train, yi_test = data.y_train.iloc[train_index], data.y_train.iloc[test_index]

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


# Test the max_depth hyperparameter

# Initialize accuracy array for plot
accuracy_by_n = []

# Initialize max_depths to test
max_depths = range(1, 21)

# Initialize best tree
best_tree = DecisionTreeClassifier()
y_train = pd.Series()
y_test = pd.Series()
X_train = pd.Series()
X_test = pd.Series()

# 10-fold cross-validation
for n in max_depths:
    decision_tree_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=n)

    # Initialize accuracy array for classifier
    accuracies = []

    for train_index, test_index in data.kf.split(data.X_train):
        Xi_train, Xi_test = data.X_train.iloc[train_index], data.X_train.iloc[test_index]
        yi_train, yi_test = data.y_train.iloc[train_index], data.y_train.iloc[test_index]

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
            X_train = Xi_train
            X_test = Xi_test
            y_train = yi_train
            y_test = yi_test

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
plot_tree(best_tree, filled=True, class_names=['survived', 'did not survive'], feature_names=data.X_train.columns, rounded=True)
plt.tight_layout()
plt.savefig('../images/decision_tree_final.png', dpi=800)
plt.show()

# Plot the ROC curve

# Predict probabilities for positive class (class 1)
y_probabilities = best_tree.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probabilities)

# Calculate AUC (Area Under the Curve)
auc = roc_auc_score(y_test, y_probabilities)
print('\nDecision Tree Classifier AUC:', auc)

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('../images/decision_tree_roc.png', dpi=800)
plt.show()


