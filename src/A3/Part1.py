import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def clean_bc():
    ##Breaset Cancer dataset
    # fetch dataset
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

    # data (as pandas dataframes)
    X = breast_cancer_wisconsin_diagnostic.data.features
    y = breast_cancer_wisconsin_diagnostic.data.targets

    # metadata
    print(breast_cancer_wisconsin_diagnostic.metadata)

    # variable information
    print(breast_cancer_wisconsin_diagnostic.variables)

    # Fetch dataset
    breast_cancer = fetch_ucirepo(id=17)
    X = breast_cancer.data.features
    y = breast_cancer.data.targets

    # Print metadata and variable information
    print("Dataset Metadata:\n", breast_cancer.metadata)
    print("\nVariable Information:\n", breast_cancer.variables)

    # Data Preprocessing

    # Check for missing values
    print("\nMissing values in features:\n", X.isnull().sum())
    print("\nMissing values in targets:\n", y.isnull().sum())

    # Display basic statistics of the features
    print("\nFeature Summary Statistics:\n", X.describe())

    # Visualize the correlation between features
    plt.figure(figsize=(12, 10))
    corr = X.corr()
    sns.heatmap(corr, cmap='coolwarm', annot=False, fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.savefig("/Users/classroomservices/PycharmProjects/ML Course/result/A3/BC_feature_corr.jpeg")
    plt.close()

    # Split the dataset into training and testing sets (80-20 split)
    X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("\nTraining set shape:", X_train_bc.shape)
    print("Testing set shape:", X_test_bc.shape)

    # Standardize the features (zero mean and unit variance)
    scaler = StandardScaler()
    X_train_bc_scaled = scaler.fit_transform(X_train_bc)
    X_test_bc_scaled = scaler.transform(X_test_bc)

    return X_train_bc_scaled, X_test_bc_scaled, y_train_bc, y_test_bc

def clean_dg():
    ##Digit Dataset
    # Load the datasets
    # Ensure that the CSV files ('train.csv' and 'test.csv') are in your working directory.
    train_df = pd.read_csv('./data/train.csv')
    test_df = pd.read_csv('./data/test.csv')

    # Display basic info about the training data
    print("Training data shape:", train_df.shape)
    print("Test data shape:", test_df.shape)
    print(train_df.head())

    # Preprocessing training data
    # Separate features and labels
    X = train_df.drop('label', axis=1).values  # all pixel columns
    y = train_df['label'].values

    # Normalize pixel values from [0, 255] to [0, 1]
    X = X.astype('float32') / 255.0

    # Optionally, reshape the flat pixel arrays to 28x28 images
    X = X.reshape(-1, 28, 28)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Display a sample image from the training set
    plt.imshow(X_train[0], cmap='gray')
    plt.title(f'Sample Image - Label: {y_train[0]}')
    plt.axis('off')
    plt.savefig("/Users/classroomservices/PycharmProjects/ML Course/result/A3/digit_sample.jpeg")

    return X_train, X_test, y_train, y_test


def train_evaluate_bc(X_train, X_test, y_train, y_test):
    """
    Trains an MLP classifier on the Breast Cancer dataset and returns evaluation metrics.

    Parameters:
        X_train, X_test, y_train, y_test: Cleaned train-test splits for features and targets.

    Returns:
        A dictionary containing 'accuracy', 'classification_report', and 'confusion_matrix'.
    """
    mlp = MLPClassifier(random_state=42, max_iter=300)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    return metrics


def train_evaluate_dg(X_train, X_test, y_train, y_test):
    """
    Trains an MLP classifier on the Digit dataset and returns evaluation metrics.

    Note: The input images are expected to be in 28x28 format. The function flattens and scales them.

    Parameters:
        X_train, X_test, y_train, y_test: Cleaned train-test splits for the Digit dataset.

    Returns:
        A dictionary containing 'accuracy', 'classification_report', and 'confusion_matrix'.
    """
    # Flatten the images (28x28 -> 784)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    mlp = MLPClassifier(random_state=42, max_iter=300)
    mlp.fit(X_train_scaled, y_train)
    y_pred = mlp.predict(X_test_scaled)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    return metrics


def print_metrics(metrics, dataset_name="Dataset"):
    """
    Prints the evaluation metrics in a readable format.

    Parameters:
        metrics (dict): Dictionary containing evaluation metrics.
        dataset_name (str): Name of the dataset for labeling the output.
    """
    print(f"\n{dataset_name} - MLP Classifier Performance:")
    print("Accuracy:", metrics['accuracy'])
    print("\nClassification Report:\n", metrics['classification_report'])
    print("Confusion Matrix:\n", metrics['confusion_matrix'])


if __name__ == '__main__':
    # For Breast Cancer dataset:
    # Get cleaned data using your cleaning function.
    X_train_bc, X_test_bc, y_train_bc, y_test_bc = clean_bc()
    bc_metrics = train_evaluate_bc(X_train_bc, X_test_bc, y_train_bc, y_test_bc)
    print_metrics(bc_metrics, "Breast Cancer Dataset")

    # For Digit dataset:
    X_train_dg, X_test_dg, y_train_dg, y_test_dg = clean_dg()
    dg_metrics = train_evaluate_dg(X_train_dg, X_test_dg, y_train_dg, y_test_dg)
    print_metrics(dg_metrics, "Digit Dataset")