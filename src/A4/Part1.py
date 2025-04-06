import os
import numpy as np
import matplotlib.pyplot as plt
import json
from keras.src.utils import to_categorical
from tensorflow.keras import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load and Preprocess the MNIST Dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Normalize pixel values to [0, 1]
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

# Reshape images to include the channel dimension (28, 28, 1)
train_images = np.expand_dims(train_images, -1)
test_images = np.expand_dims(test_images, -1)

# Create a 5x5 grid to display 25 sample training images with their labels.
fig, axs = plt.subplots(5, 5, figsize=(10, 10))
for i, ax in enumerate(axs.flat):
    if i < len(train_images):
        ax.imshow(train_images[i], cmap="gray")
        ax.set_title(f"Label: {train_labels[i]}")
        ax.axis("off")
    else:
        ax.axis("off")

plt.tight_layout()
grid_filename = "mnist_sample_grid.png"
plt.savefig(grid_filename)
plt.close()
print(f"Sample grid figure saved as {grid_filename}")

# Visualization 2: Histogram of Label Distribution
plt.figure(figsize=(8, 6))
bins = np.arange(11) - 0.5  # Create bins for digits 0-9
plt.hist(train_labels, bins=bins, rwidth=0.8, color='skyblue', edgecolor='black')
plt.title("Distribution of Digit Labels (Training Set)")
plt.xlabel("Digit")
plt.ylabel("Frequency")
plt.xticks(np.arange(10))
plt.grid(axis='y', alpha=0.75)

hist_filename = "mnist_label_distribution.png"
plt.savefig(hist_filename)
plt.close()
print(f"Label distribution figure saved as {hist_filename}")


# One-hot encode the labels
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# Define Hyperparameter Grid
# 3 filter sizes, 2 optimizers, 2 conv_layer options, 3 dense neurons options
filter_sizes = [16, 32, 64]  # base number of filters for the first conv layer
optimizers = ['adam', 'sgd']
conv_layers_options = [1, 2]  # using either 1 or 2 convolutional layers
dense_neurons_options = [64, 128, 256]  # number of neurons in the dense layer

# Define Early Stopping Callback
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


# Function to Build a CNN Model
def build_model(optimizer, conv_layers, filter_size, dense_neurons, dropout_rate=0.25):
    model = Sequential()
    input_shape = (28, 28, 1)

    # Add first convolutional layer (with input shape defined)
    model.add(Conv2D(filter_size, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

    if conv_layers == 2:
        # For 2 conv layers, add a second conv layer.
        # Here we use double the filter size for the second layer for variety.
        model.add(Conv2D(filter_size * 2, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    else:
        # For a single conv layer, simply add max pooling.
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(dense_neurons, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Training Loop Over All Configurations
results = {}
config_count = 0
total_configs = len(filter_sizes) * len(optimizers) * len(conv_layers_options) * len(dense_neurons_options)

epochs = 1
batch_size = 128

for optimizer in optimizers:
    for conv_layers in conv_layers_options:
        for filter_size in filter_sizes:
            for dense_neurons in dense_neurons_options:
                config_count += 1
                model_name = f"model_{optimizer}_{conv_layers}conv_{filter_size}f_{dense_neurons}d"
                print(f"Training {model_name} ({config_count}/{total_configs})...")

                model = build_model(optimizer, conv_layers, filter_size, dense_neurons)
                history = model.fit(train_images, train_labels,
                                    validation_split=0.1,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    callbacks=[early_stop],
                                    verbose=1)

                # Evaluate on the test set
                test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)

                # Save configuration and performance metrics
                results[model_name] = {
                    'optimizer': optimizer,
                    'conv_layers': conv_layers,
                    'filter_size': filter_size,
                    'dense_neurons': dense_neurons,
                    'dropout_rate': 0.25,
                    'epochs_trained': len(history.history['loss']),
                    'test_loss': test_loss,
                    'test_accuracy': test_accuracy,
                    'history': history.history
                }

                print(f"{model_name} -- Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}\n")

# Save All Results to a JSON File
with open("cnn_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("All models trained and results saved to cnn_results.json")