from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import re
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv('urbansounds_features.csv')


def calculate_mean_from_string(string):
    cleaned_string = string.replace('\n', '')
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", cleaned_string)
    array = np.array(numbers, dtype=float)
    mean_value = np.mean(array)
    return mean_value


# df['tempogram'] = df['tempogram'].apply(calculate_mean_from_string)
df['fourier_tempogram'] = df['fourier_tempogram'].apply(calculate_mean_from_string)
df['Label'] = df['Label'].str.split('-').str[1]

X = df.drop('Label', axis=1)
y = df['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Oversample the features values using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Standardize the feature values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Calculate the mean number of neurons for hidden layers
mean_neurons = (X_train_scaled.shape[1] + len(np.unique(y_train_resampled))) // 2

# Define the hyperparameters to search
parameters = {
    'hidden_layer_sizes': [(mean_neurons,), (mean_neurons, mean_neurons), (mean_neurons, mean_neurons, mean_neurons)],
    'activation': ['relu', 'softmax', 'step', 'sigmoid'],
    'max_iter': [300]
}

# Create an MLP classifier
mlp_classifier = MLPClassifier()

# Use GridSearchCV to search for the best hyperparameters
grid_search = GridSearchCV(mlp_classifier, parameters, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train_resampled)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Create the final MLP classifier with the best hyperparameters
final_mlp_classifier = MLPClassifier(**best_params)
final_mlp_classifier.fit(X_train_scaled, y_train_resampled)
y_pred = final_mlp_classifier.predict(X_test_scaled)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Best Hyperparameters: {best_params}')
print(f'Accuracy: {accuracy}')
