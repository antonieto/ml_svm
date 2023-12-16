import encoding as data
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Initialize accuracy array for plot
accuracy_by_k = []

# 10-fold cross-validation
for k in range(1, 21):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)

    # Initialize accuracy array for classifier
    accuracies = []

    for train_index, test_index in data.kf.split(data.X):
        Xi_train, Xi_test = data.X.iloc[train_index], data.X.iloc[test_index]
        yi_train, yi_test = data.y.iloc[train_index], data.y.iloc[test_index]

        # Fit the model to the training data
        knn_classifier.fit(Xi_train, yi_train)

        # Predict probabilities for each class for the test data
        yi_pred = knn_classifier.predict(Xi_test)

        # Calculate accuracy and append to the accuracies list
        accuracy = accuracy_score(yi_test, yi_pred)
        accuracies.append(accuracy)

    accuracy_by_k.append(np.mean(accuracies))

    print('\nNearest Neighbor Classifier k =', k, ':')
    print('Accuracy =', '{:.2f} %'.format(round(np.mean(accuracies) * 100, 2)))
    print('Std\t\t =', '{:.4f}'.format(round(np.std(accuracies), 4)))

plt.bar(list(range(1, 21)), accuracy_by_k, color='skyblue')
plt.title('Accuracy vs. Number of Neighbors (k)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(list(range(1, 21)))
plt.ylim(0.8, 0.95)
plt.savefig('../images/nearest_neighbor_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()
