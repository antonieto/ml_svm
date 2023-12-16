import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer

import encoding as data
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, RocCurveDisplay
from sklearn.naive_bayes import MultinomialNB

# Create the Multinomial Naive Bayes model
naive_bayes_classifier = MultinomialNB()

# Initialize accuracies array
accuracies = []

# Initialize best classifier
best_classifier = naive_bayes_classifier
y_train = pd.Series()
y_test = pd.Series()
X_train = pd.Series()
X_test = pd.Series()

# 10-fold cross-validation
for train_index, test_index in data.kf.split(data.X):
    Xi_train, Xi_test = data.X.iloc[train_index], data.X.iloc[test_index]
    yi_train, yi_test = data.y.iloc[train_index], data.y.iloc[test_index]

    # Fit the model to the training data
    naive_bayes_classifier.fit(Xi_train, yi_train)

    # Predict probabilities for each class for the test data
    yi_pred = naive_bayes_classifier.predict(Xi_test)

    # Calculate accuracy and append to the accuracies list
    accuracy = accuracy_score(yi_test, yi_pred)
    accuracies_before = list(accuracies)
    accuracies.append(accuracy)

    if accuracies_before != [] and accuracy > max(accuracies_before):
        best_classifier = naive_bayes_classifier
        X_train = Xi_train
        X_test = Xi_test
        y_train = yi_train
        y_test = yi_test


# Calculate and print results: accuracy mean and standard deviation
print('Naive Bayes Classifier:')
print('Accuracy =', '{:.2f} %'.format(round(np.mean(accuracies) * 100, 2)))
print('Std\t\t =', '{:.4f}'.format(round(np.std(accuracies), 4)))

# Plot the ROC curves for the best classifier
# Determine class ids for every class
label_binarizer = LabelBinarizer().fit(y_train.to_numpy())
class_id_value = {}
for c in data.class_order:
    class_id = data.encoder.categories_[len(data.encoder.categories_)-1].tolist().index(c)
    class_id_value[class_id] = c

for class_id in class_id_value.keys():
    classname = class_id_value[class_id]
    y_score = naive_bayes_classifier.fit(X_train, y_train).predict_proba(X_test)
    y_onehot_test = label_binarizer.transform(y_test.to_numpy())

    RocCurveDisplay.from_predictions(
        y_onehot_test[:, class_id],
        y_score[:, class_id],
        name=f"{classname} vs the rest",
        color="darkorange",
        plot_chance_level=True,
    )

    rest = [element for element in class_id_value.values() if element != classname]
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"One-vs-Rest ROC curves:\n\'{classname}\' vs {rest}")
    plt.legend()
    plt.savefig(f'../images/roc_naive_bayes_{classname}')
    plt.show()
