# Multi layer perceptron

from data import kf, X_train, y_train
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.neural_network import MLPClassifier


# Create MLP classifier
best_mlp = None

# Define parameter grid to search
param_grid = {
    'hidden_layer_sizes': [(50, 50), (100, 50)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam'],
    'learning_rate': ['constant'],
    'alpha': [0.01, 0.001],
    'batch_size': [64, 128],
    'max_iter': [500]
}

layer_sizes = [(256, ), (512,), (512, 256), (512, 512)]
best_sizes = None

# Accuracies for each size choice
size_accuracies = []

for sizes in layer_sizes:
  accuracies = []
  # Fit the model to the training data
  mlp = MLPClassifier(
      activation='logistic',
      hidden_layer_sizes=sizes,
      solver='adam',
      learning_rate='invscaling',
      alpha=0.001,
      batch_size=64,
      max_iter=600
  )
  for train_index, test_index in kf.split(X_train):
        Xi_train, Xi_test = X_train.iloc[train_index], X_train.iloc[test_index]
        yi_train, yi_test = y_train.iloc[train_index], y_train.iloc[test_index]

        # Fit the model to the training data
        mlp.fit(Xi_train, yi_train)

        # Predict probabilities for each class for the test data
        yi_pred = mlp.predict(Xi_test)

        # Calculate accuracy and append to the accuracies list
        accuracy = accuracy_score(yi_test, yi_pred)
        accuracies_before = list(accuracies)
        accuracies.append(accuracy)

        # Update best tree if accuracy is biggest
        if accuracies_before != [] and accuracy > max(accuracies_before):
          best_sizes = sizes
          best_mlp = mlp

  size_accuracies.append(np.mean(accuracies))
  # Calculate and print results: accuracy mean and standard deviation
  print('Accuracy( layer sizes = ', sizes, ') =', '{:.2f} %'.format(round(np.mean(accuracies) * 100, 2)))
  print('Std( sizes =', sizes, ')\t\t =', '{:.4f}'.format(round(np.std(accuracies), 4)))

print(f'Best sizes = {best_sizes}')

# Choose solver
activation_choices = ['identity', 'logistic', 'tanh', 'relu']

activation_accuracies = []

for activation in activation_choices:
  mlp = MLPClassifier(
      activation=activation,
      hidden_layer_sizes=best_sizes,
      solver='adam',
      learning_rate='invscaling',
      alpha=0.001,
      batch_size=64,
      max_iter=600
  )
  accuracies = []
  for train_index, test_index in kf.split(X_train):
    Xi_train, Xi_test = X_train.iloc[train_index], X_train.iloc[test_index]
    yi_train, yi_test = y_train.iloc[train_index], y_train.iloc[test_index]

    # Fit the model to the training data
    mlp.fit(Xi_train, yi_train)

    # Predict probabilities for each class for the test data
    yi_pred = mlp.predict(Xi_test)

    # Calculate accuracy and append to the accuracies list
    accuracy = accuracy_score(yi_test, yi_pred)
    accuracies_before = list(accuracies)
    accuracies.append(accuracy)

    # Update best tree if accuracy is biggest
    if accuracies_before != [] and accuracy > max(accuracies_before):
      best_activation = activation
      best_mlp = mlp  

  print('Accuracy( activation function = ', activation, ') =', '{:.2f} %'.format(round(np.mean(accuracies) * 100, 2)))
  print('Std( activation function =', activation, ')\t\t =', '{:.4f}'.format(round(np.std(accuracies), 4)))

# Plot the ROC curve

# Predict probabilities for positive class (class 1)
y_probabilities = best_mlp.predict_proba(X_test)[:, 1]

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
plt.show()

