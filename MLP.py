import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import accuracy_score
import re


def calculate_mean_from_string(string):
    cleaned_string = string.replace('\n', '')
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", cleaned_string)
    array = np.array(numbers, dtype=float)
    mean_value = np.mean(array)
    return mean_value


# Assuming you have 10 datasets: 'urbansounds_features_1.csv' to 'urbansounds_features_10.csv'
datasets = [pd.read_csv(f'datasets/urbansounds_features_{i}.csv') for i in range(1, 11)]

for df in datasets:
    for column in df.columns:
        if column != 'Label':
            if df[column].dtype != float and df[column].dtype != int:
                df[column] = df[column].apply(calculate_mean_from_string)
        else:
            df[column] = df[column].str.split('-').str[1].astype(int)

# Hyperparameters
dropout = 0.2
activation_function = 'relu'
loss_function = 'sparse_categorical_crossentropy'
batch_size = 64
num_epochs = 300
learning_rate = 0.015

# Create empty list to store results
cv_scores = []

# Set up k-fold cross-validation
stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

all_labels = np.concatenate([df['Label'].values for df in datasets])


for fold, (train_index, val_index) in enumerate(stratified_kfold.split(range(len(all_labels)), all_labels)):
    # Use the current fold as the validation set
    validation_dataset = datasets[fold]

    # Combine the remaining datasets as the training set
    training_datasets = [dataset for index, dataset in enumerate(datasets) if index != fold]
    combined_df = pd.concat(training_datasets, ignore_index=True)

    # Classification
    X_train = combined_df.drop('Label', axis=1)
    y_train = combined_df['Label']

    # Oversample the features values using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Standardize the feature values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_resampled)

    # Classification for validation set
    X_val = validation_dataset.drop('Label', axis=1)
    y_val = validation_dataset['Label']

    # Oversample the features values using SMOTE for validation set
    X_val_resampled, y_val_resampled = smote.fit_resample(X_val, y_val)
    X_val_scaled = scaler.transform(X_val_resampled)

    mean_neurons = (X_train_scaled.shape[1] + len(np.unique(y_resampled))) // 2
    num_input_neurons = X_train_scaled.shape[1]
    num_output_neurons = len(np.unique(y_resampled))
    neurons_hidden_layer = int(2 / 3 * num_input_neurons + 1 / 3 * num_output_neurons)

    # Define and compile the model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=neurons_hidden_layer, activation=activation_function,
                              input_shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(units=mean_neurons, activation=activation_function),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(units=len(np.unique(y_resampled)), activation='softmax')
    ])
    model.compile(loss=loss_function, optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate))

    # Train the model
    model.fit(X_train_scaled, y_resampled, epochs=num_epochs, validation_data=(X_val_scaled, y_val_resampled),
              batch_size=batch_size)

    # Evaluate the model on the validation set
    y_val_pred_probs = model.predict(X_val_scaled)
    y_val_pred = np.argmax(y_val_pred_probs, axis=1)

    # Calculate and store accuracy for this fold
    fold_accuracy = accuracy_score(y_val_resampled, y_val_pred)
    cv_scores.append(fold_accuracy)

# Calculate and print the overall average accuracy
overall_average_accuracy = np.mean(cv_scores)
print(f'Overall Average Cross-Validation Accuracy: {overall_average_accuracy}')
