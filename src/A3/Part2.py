import json
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from Part1 import clean_dg, clean_bc

# Define hyper-parameter options
# Four network structures: The second element of the tuple is used as hidden_layer_sizes for MLPClassifier.
structures = [
    (2, (64, 64)),
    (2, (128, 128)),
    (3, (64, 64, 64)),
    (3, (128, 128, 128))
]
# Activation functions: "logistic" is the sigmoid equivalent in scikit-learn.
activations = ['logistic', 'tanh', 'relu']
# Optimizers (solvers)
optimizers = ['adam', 'sgd']

results = []

# Get cleaned data for both datasets
X_train_bc, X_test_bc, y_train_bc, y_test_bc = clean_bc()
X_train_dg, X_test_dg, y_train_dg, y_test_dg = clean_dg()

# --- Experiments on Breast Cancer Dataset ---
for struct in structures:
    hidden_layer_sizes = struct[1]
    for activation in activations:
        for optimizer in optimizers:
            # Initialize and train the MLP classifier for Breast Cancer data
            clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                activation=activation,
                                solver=optimizer,
                                random_state=42,
                                max_iter=300)
            clf.fit(X_train_bc, y_train_bc)
            y_pred = clf.predict(X_test_bc)

            # Evaluate performance metrics
            acc = accuracy_score(y_test_bc, y_pred)
            report = classification_report(y_test_bc, y_pred, output_dict=True)
            cm = confusion_matrix(y_test_bc, y_pred)

            # Save the results
            result = {
                'dataset': 'Breast Cancer',
                'hidden_layers': len(hidden_layer_sizes),
                'neurons': hidden_layer_sizes,
                'activation': activation,
                'optimizer': optimizer,
                'accuracy': acc,
                'classification_report': report,
                'confusion_matrix': cm.tolist()  # convert numpy array to list for JSON serialization
            }
            results.append(result)

# --- Experiments on Digit Dataset ---
# For the digit dataset, we flatten and standardize the image data.
for struct in structures:
    hidden_layer_sizes = struct[1]
    for activation in activations:
        for optimizer in optimizers:
            # Flatten images (from 28x28 to 784)
            X_train_flat = X_train_dg.reshape(X_train_dg.shape[0], -1)
            X_test_flat = X_test_dg.reshape(X_test_dg.shape[0], -1)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_flat)
            X_test_scaled = scaler.transform(X_test_flat)

            # Initialize and train the MLP classifier for Digit data
            clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                activation=activation,
                                solver=optimizer,
                                random_state=42,
                                max_iter=300)
            clf.fit(X_train_scaled, y_train_dg)
            y_pred = clf.predict(X_test_scaled)

            # Evaluate performance metrics
            acc = accuracy_score(y_test_dg, y_pred)
            report = classification_report(y_test_dg, y_pred, output_dict=True)
            cm = confusion_matrix(y_test_dg, y_pred)

            # Save the results
            result = {
                'dataset': 'Digit',
                'hidden_layers': len(hidden_layer_sizes),
                'neurons': hidden_layer_sizes,
                'activation': activation,
                'optimizer': optimizer,
                'accuracy': acc,
                'classification_report': report,
                'confusion_matrix': cm.tolist()
            }
            results.append(result)

# Create a pandas DataFrame to display the results in a table
df_results = pd.DataFrame(results)
print(df_results)

# Save the complete results to a JSON file
with open('mlp_experiments_results.json', 'w') as f:
    json.dump(results, f, indent=4)
