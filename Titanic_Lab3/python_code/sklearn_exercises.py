import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import RocCurveDisplay, roc_curve, auc
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, LabelBinarizer
from sklearn.naive_bayes import MultinomialNB

'''
Loading, Preprocessing, and Split of Data Frame
'''

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

'''
Task 1 -- Classification with Naive Bayes Classifier
'''

# Create the Multinomial Naive Bayes model
naive_bayes_classifier = MultinomialNB()

# Initialize arrays to store TPR and FPR values for each fold
all_tpr = []
all_fpr = []
accuracies = []

for train_index, test_index in kf.split(X):
    # Split the data according to 10-fold
    Xi_train, Xi_test = X.iloc[train_index], X.iloc[test_index]
    yi_train, yi_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the model to the training data and compute accuracy on the testing data
    naive_bayes_classifier.fit(Xi_train, yi_train)

    # Save accuracy
    accuracy = naive_bayes_classifier.score(Xi_test, yi_test)
    accuracies.append(accuracy)

    y_score = naive_bayes_classifier.fit(Xi_train, yi_train).predict_proba(Xi_test)
    label_binarizer = LabelBinarizer().fit(yi_train.to_numpy())
    y_onehot_test = label_binarizer.transform(yi_test.to_numpy())

    class_of_interest = encoder.categories_[len(encoder.categories_)-1].tolist().index('unacc')
    class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
    fpr, tpr, thresholds = roc_curve(y_onehot_test[:, class_id], y_score[:, class_id])
    print(len(fpr))

    all_fpr.append(fpr)
    all_tpr.append(tpr)

    '''RocCurveDisplay.from_predictions(
        y_onehot_test[:, class_id],
        y_score[:, class_id],
        name=f"{class_of_interest} vs the rest",
        color="darkorange",
        plot_chance_level=True,
    )
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-Rest ROC curves:\nVirginica vs (Setosa & Versicolor)")
    plt.legend()
    plt.show()'''


# Calculate and print the mean accuracy and standard deviation
mean_accuracy = sum(accuracies) / len(accuracies)
print(mean_accuracy)

# Calculate mean fpr and mean tpr
mean_fpr = [sum(x) / len(x) for x in zip(*all_fpr)]
mean_tpr = [sum(x) / len(x) for x in zip(*all_tpr)]
for element in all_fpr:
    print(len(element))
roc_auc = auc(np.array(mean_fpr).ravel(), np.array(mean_tpr).ravel())

# Create a figure and axis
plt.figure(figsize=(6, 6))

# Plot the ROC curve
plt.plot(mean_fpr, mean_tpr, color='darkorange', lw=2, label=f"0 vs the rest (AUC = {roc_auc:.2f})")

# Plot the diagonal line representing chance-level performance
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', label='Chance level (AUC = 0.5)')

# Set labels and title
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('One-vs-Rest ROC curve:\n unacc vs rest')

# Show the legend
plt.legend(loc='lower right')

# Show the plot
plt.show()

# average measures after 10 fold: accuracy, roc extension for multiclass (microaverage, macroaverage), multiclass, randindex

