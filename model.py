import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO logs

import tensorflow as tf

# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to avoid pre-allocating all GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU is available and memory growth is enabled.")
        print("Using GPU:", tf.config.list_logical_devices('GPU'))
    except RuntimeError as e:
        print("❌ RuntimeError while setting GPU configuration:", e)
else:
    print("⚠️ No GPU found. Using CPU.")
import json
import matplotlib.pyplot as plt
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import fashion_mnist, cifar10, cifar100
import tensorflow.keras.backend as K
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


     

# Define folder structure
BASE_DIR = "saved_models"
PLOT_DIR = "training_graphs"
RESULTS_DIR = "results"

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


     

# Custom activation functions
def penalized_tanh(x, alpha=0.25):  # alpha is a hyperparameter (default 0.25)
    return tf.where(x >= 0, tf.tanh(x), alpha * tf.tanh(x))

def eliSH(x):
    return x * tf.nn.sigmoid(x) * (1 + tf.nn.tanh(x))

def mish(x):
    return x * tf.nn.tanh(K.softplus(x))  # softplus(x) = log(1 + exp(x))

def rsigelu(x):
    return x * tf.nn.sigmoid(x) + tf.nn.elu(x)

def tanh_exp(x):
    return x * tf.tanh(K.exp(x))

def pflu(x, alpha=0.1, beta=1.0):
    return tf.where(x >= 0, x + beta, alpha * (tf.exp(x) - 1))

def gcu(x):
    return x * tf.cos(x)

def hcLSH(x):
    return x * tf.nn.sigmoid(x) * K.log(K.sigmoid(x) + K.epsilon())


     

activation_functions = {
    "relu": tf.nn.relu,
    "tanh": tf.nn.tanh,
    "sigmoid": tf.nn.sigmoid,
    "hcLSH": hcLSH,
    "penalized_tanh": penalized_tanh,
    "eliSH": eliSH,
    "mish": mish,
    "rsigelu": rsigelu,
    "tanh_exp": tanh_exp,
    "pflu": pflu,
    "gcu": gcu
}


     

activation_colors = {
    "relu": "blue",
    "tanh": "green",
    "sigmoid": "red",
    "hcLSH": "purple",
    "penalized_tanh": "orange",
    "eliSH": "cyan",
    "mish": "magenta",
    "rsigelu": "brown",
    "tanh_exp": "pink",
    "pflu": "gray",
    "gcu": "olive"
}


     

def load_svhn():
    train_data = loadmat("path/to/train_32x32.mat")
    test_data = loadmat("path/to/test_32x32.mat")

    X_train = np.moveaxis(train_data["X"], -1, 0)  # Move axis to match TF format
    y_train = train_data["y"].flatten() % 10  # Normalize labels (SVHN uses 1-10)
    
    X_test = np.moveaxis(test_data["X"], -1, 0)
    y_test = test_data["y"].flatten() % 10

    return X_train, X_test, y_train, y_test


     

def load_dataset(dataset_name):
    if dataset_name == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        X_train = X_train[..., np.newaxis]  # Add channel dimension
        X_test = X_test[..., np.newaxis]

    elif dataset_name == "cifar10":
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    elif dataset_name == "cifar100":
        (X_train, y_train), (X_test, y_test) = cifar100.load_data()

    elif dataset_name == "svhn":
        X_train, X_test, y_train, y_test = load_svhn()

    else:
        raise ValueError("Dataset not supported!")

    # Normalize data
    X_train, X_test = X_train / 255.0, X_test / 255.0

    return X_train, X_test, y_train, y_test


     

def create_cnn_model(input_shape, activation_name, num_classes):
    activation_func = activation_functions[activation_name]

    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), padding="same", input_shape=input_shape),
        BatchNormalization(),
        Lambda(activation_func),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=64, kernel_size=(3, 3), padding="same"),
        BatchNormalization(),
        Lambda(activation_func),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256),
        Lambda(activation_func),
        Dropout(0.3),     
        Dense(128),
        Lambda(activation_func),
        Dropout(0.4),
        Dense(num_classes, activation="softmax")  # Multi-class classification
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
     

def create_rnn_model(activation_name, input_shape, num_classes):
    activation_fn = activation_functions[activation_name]  # Default to Tanh if not found

    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(64, activation=activation_fn, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.SimpleRNN(128, activation=activation_fn, return_sequences=False),
        tf.keras.layers.Dense(64, activation=activation_fn),
        tf.keras.layers.Dense(128, activation=activation_fn),
        tf.keras.layers.Dense(num_classes, activation="softmax")  # Output layer
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


     

# Function to save model properly
def save_model(model, dataset_name, activation_name):
    model_dir = os.path.join("saved_models", dataset_name)
    os.makedirs(model_dir, exist_ok=True)  # Ensure directory exists

    # Save model weights (only weights, NOT full model)
    model_path = os.path.join(model_dir, f"cnn_{dataset_name}_{activation_name}.h5")
    model.save_weights(model_path)  # Save weights
    print(f"✅ Model weights saved: {model_path}")

    # Save metadata separately (since model structure cannot be serialized)
    metadata = {
        "dataset": dataset_name,
        "activation": activation_name,
    }
    metadata_path = os.path.join(model_dir, f"metadata_{dataset_name}_{activation_name}.json")
    with open(metadata_path, "w") as json_file:
        json.dump(metadata, json_file, indent=4)

    print(f"✅ Metadata saved: {metadata_path}")


     

# Function to reload the model
def load_saved_model(dataset_name, activation_name, input_shape, num_classes):
    model_dir = os.path.join("saved_models", dataset_name)

    # Rebuild the model
    model = create_cnn_model(input_shape, activation_name, num_classes)

    # Load saved weights
    model_path = os.path.join(model_dir, f"cnn_{dataset_name}_{activation_name}.h5")
    model.load_weights(model_path)
    print(f"✅ Model weights loaded: {model_path}")

    return model


     

# Function to plot training results
def plot_training(history, dataset_name, activation_name):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    acc = [a * 100 for a in history.history["accuracy"]]
    val_acc = [a * 100 for a in history.history["val_accuracy"]]

    loss= [a * 1 for a in history.history["loss"]]
    val_loss= [a * 1 for a in history.history["val_loss"]]
    
    epochs=range(len(acc))

    # Accuracy Plot
    ax[0].plot(epochs,acc, label="Train Accuracy")
    ax[0].plot(epochs,val_acc, label="Validation Accuracy")
    ax[0].set_title(f"Accuracy ({dataset_name}, {activation_name})")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()

    # Loss Plot
    ax[1].plot(epochs,loss, label="Train Loss")
    ax[1].plot(epochs,val_loss, label="Validation Loss")
    ax[1].set_title(f"Loss ({dataset_name}, {activation_name})")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    ax[1].legend()

    # Save plot
    plot_filename = f"{PLOT_DIR}/training_{dataset_name}_{activation_name}.png"
    plt.savefig(plot_filename)
    plt.close()
    print(f"📊 Plot saved as {plot_filename}")


     

# Function to save test results per dataset
def save_test_results(dataset_name, activation_name, test_loss, test_acc):
    results_file = os.path.join(RESULTS_DIR, f"{dataset_name}_results.csv")
    
    file_exists = os.path.isfile(results_file)

    with open(results_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Activation Function", "Test Loss", "Test Accuracy"])  # Header
        writer.writerow([activation_name, test_loss, test_acc])  # Write results
    
    print(f"✅ Test accuracy saved in {results_file}")


     

datasets = ["fashion_mnist", "cifar10", "cifar100"] 
activation_functions = {
    "relu": tf.nn.relu,
    "tanh": tf.nn.tanh,
    "sigmoid": tf.nn.sigmoid,
    "hcLSH": hcLSH,
    "penalized_tanh": penalized_tanh,
    "eliSH": eliSH,
    "mish": mish,
    "rsigelu": rsigelu,
    "tanh_exp": tanh_exp,
    "gcu": gcu
}
for dataset in datasets:
        for activation in activation_functions:
            print(f"🔹 Training on {dataset} with activation: {activation}")
            
            X_train, X_test, y_train, y_test = load_dataset(dataset)
            num_classes = len(np.unique(y_train))
            
            model = create_cnn_model(X_train.shape[1:], activation, num_classes)
            model.summary()
            
            history=model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
            
            save_model(model, dataset, activation)
            plot_training(history, dataset, activation)
            
            test_loss, test_acc = model.evaluate(X_test, y_test)
            print(f"📊 Test Accuracy on {dataset} with {activation}: {test_acc:.4f}")
            
            # Save test accuracy results
            save_test_results(dataset, activation, test_loss, test_acc)