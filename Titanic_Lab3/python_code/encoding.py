import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder

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
