import pandas as pd
import data as data
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, RocCurveDisplay, roc_auc_score
from sklearn.svm import SVC


"""
Usando el clasificador SVM, ajustando los hiperparametros 'kernel' y 'C',
se puede obtener un accuracy de 75.44% con el kernel 'linear' y un C de 0.1.
El AUC es 0.9231.
A continuación vemos el entrenamiento.
"""


# Initialize kernels to be tested
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

# Initialize dictionaries to store accuracies
accuracies_mean = {}
accuracies_std = {}

# Test the criterion values
for kernel in kernels:
    # Create the Multinomial Naive Bayes model
    svm_classifier = SVC(kernel=kernel, random_state=42)

    # Initialize accuracies array
    accuracies = []

    # 10-fold cross-validation
    for train_index, test_index in data.kf.split(data.X_train):
        Xi_train, Xi_test = data.X_train.iloc[train_index], data.X_train.iloc[test_index]
        yi_train, yi_test = data.y_train.iloc[train_index], data.y_train.iloc[test_index]

        # Fit the model to the training data
        svm_classifier.fit(Xi_train, yi_train)

        # Predict probabilities for each class for the test data
        yi_pred = svm_classifier.predict(Xi_test)

        # Calculate accuracy and append to the accuracies list
        accuracy = accuracy_score(yi_test, yi_pred)
        accuracies_before = list(accuracies)
        accuracies.append(accuracy)

    # Store mean and standard deviation of accuracies
    accuracies_mean[kernel] = np.mean(accuracies) * 100
    accuracies_std[kernel] = np.std(accuracies)

    # Calculate and print results: accuracy mean and standard deviation
    print('SVM Classifier:\n')
    print('Accuracy( kernel = ', kernel, ') =', '{:.2f} %'.format(round(accuracies_mean[kernel], 2)))
    print('Std( kernel =', kernel, ')\t\t =', '{:.4f}'.format(round(accuracies_std[kernel], 4)))


# Plot bar chart
fig, ax = plt.subplots()
kernels_labels = [f'{kernel}\n({accuracies_mean[kernel]:.2f}%)' for kernel in kernels]
bars = ax.bar(kernels, accuracies_mean.values(), yerr=accuracies_std.values(), capsize=8, color='skyblue')

# Add labels and title
ax.set_ylabel('Accuracy (%)')
ax.set_title('SVM Classifier Accuracies by Kernel')
ax.set_xticks(kernels)
ax.set_xticklabels(kernels_labels)

# Add values on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

# Show the plot
plt.savefig('../images/svm_kernel_accuracies.png')
plt.show()


# Test the c hyperparameter

# Initialize accuracy array for plot
accuracy_by_c = []

# Initialize max_depths to test
cs = np.logspace(-2, 1, 40)

# Initialize best classifer
best_classifier = SVC(kernel='linear', C=0.01, probability=True)
y_train = pd.Series()
y_test = pd.Series()
X_train = pd.Series()
X_test = pd.Series()

# 10-fold cross-validation
for c in cs:
    svm_classifier = SVC(kernel='linear', C=c, probability=True)

    # Initialize accuracy array for classifier
    accuracies = []

    for train_index, test_index in data.kf.split(data.X_train):
        Xi_train, Xi_test = data.X_train.iloc[train_index], data.X_train.iloc[test_index]
        yi_train, yi_test = data.y_train.iloc[train_index], data.y_train.iloc[test_index]

        # Fit the model to the training data
        svm_classifier.fit(Xi_train, yi_train)

        # Predict probabilities for each class for the test data
        yi_pred = svm_classifier.predict(Xi_test)

        # Calculate accuracy and append to the accuracies list
        accuracy = accuracy_score(yi_test, yi_pred)
        accuracies_before = list(accuracies)
        accuracies.append(accuracy)

        # Update best classifer if accuracy is biggest
        if accuracies_before != [] and accuracy > max(accuracies_before):
            best_classifier = svm_classifier
            X_train = Xi_train
            X_test = Xi_test
            y_train = yi_train
            y_test = yi_test

    accuracy_by_c.append(np.mean(accuracies))

    print('\nSVM Classifier C =', c, ':')
    print('Accuracy =', '{:.2f} %'.format(round(np.mean(accuracies) * 100, 2)))
    print('Std\t\t =', '{:.4f}'.format(round(np.std(accuracies), 4)))

# plot Accuracy vs. Maximum Depth
plt.plot(cs, accuracy_by_c)
plt.title('Accuracy vs. C')
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.xticks([0.01, 0.1, 1, 10])
plt.tight_layout()
plt.savefig('../images/svm_accuracy_c.png', dpi=300)
plt.show()

# Plot the ROC curve

# Predict probabilities for positive class (class 1)
y_probabilities = best_classifier.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probabilities)

# Calculate AUC (Area Under the Curve)
auc = roc_auc_score(y_test, y_probabilities)
print('\nSVM Classifier AUC:', auc)

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('../images/svm_roc.png', dpi=800)
plt.show()
