import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from RandomForest import RandomForest, DecisionTree
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score

np.set_printoptions(threshold=10000, linewidth=2000)

# Fetch dataset
letter_recognition = fetch_ucirepo(id=59)

# Data (as pandas dataframes)
X: pd.DataFrame = letter_recognition.data.features
y = letter_recognition.data.targets

print(X.dtypes)
print(y.dtypes)

# Checking missing values
print(X.isna().sum())

# Encoding labels (if necessary)
encoder = LabelEncoder()
y = encoder.fit_transform(y)  # Apply LabelEncoder to the target variable (labels)

# Split data into train and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features (standardizing to have mean=0 and variance=1)
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# Initialize the classifiers
dt_classifier = DecisionTree(max_depth=10)
rf_classifier = RandomForest(n_trees=100, max_depth=15)


# Train the Decision Tree model
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)

# Train the Random Forest model
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)

# Evaluate the models
# Decision Tree
one_hot_encoder = OneHotEncoder(sparse_output=False)
encoded_y_test = one_hot_encoder.fit_transform(y_test.reshape(-1, 1))
encoded_y_pred_dt = one_hot_encoder.transform(y_pred_dt.reshape(-1, 1))
encoded_y_pred_rf = one_hot_encoder.transform(y_pred_rf.reshape(-1, 1))

print("Decision Tree Classifier Metrics:")
print(f"Accuracy: {accuracy_score(encoded_y_test, encoded_y_pred_dt)}")
print(f"ROC AUC: {roc_auc_score(encoded_y_test, encoded_y_pred_dt, multi_class='ovr')}")
print(f"Precision: {precision_score(encoded_y_test, encoded_y_pred_dt, average='weighted')}")
print(f"Recall: {recall_score(encoded_y_test, encoded_y_pred_dt, average='weighted')}")
print(f"F1 Score: {f1_score(encoded_y_test, encoded_y_pred_dt, average='weighted')}")
print(f"Confusion Matrix:\n{np.array(confusion_matrix(y_test, y_pred_dt))}\n")

# Random Forest
print("Random Forest Classifier Metrics:")
print(f"Accuracy: {accuracy_score(encoded_y_test, encoded_y_pred_rf)}")
print(f"ROC AUC: {roc_auc_score(encoded_y_test, encoded_y_pred_rf, multi_class='ovr')}")
print(f"Precision: {precision_score(encoded_y_test, encoded_y_pred_rf, average='weighted')}")
print(f"Recall: {recall_score(encoded_y_test, encoded_y_pred_rf, average='weighted')}")
print(f"F1 Score: {f1_score(encoded_y_test, encoded_y_pred_rf, average='weighted')}")
print(f"Confusion Matrix:")
print(np.array(confusion_matrix(y_test, y_pred_rf)))
