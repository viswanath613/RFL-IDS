import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random

# ===============================
# 1. Load Dataset
# ===============================
def load_data(path):
    df = pd.read_csv(path)

    # Assume last column = label
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    return train_test_split(X, y, test_size=0.2, random_state=42)

# ===============================
# 2. Create CNN Model
# ===============================
def create_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# ===============================
# 3. Split Data into Clients
# ===============================
def create_clients(X, y, num_clients=10):
    data = list(zip(X, y))
    random.shuffle(data)

    split_size = len(data) // num_clients
    clients = []

    for i in range(num_clients):
        subset = data[i*split_size:(i+1)*split_size]
        X_c = np.array([x for x, _ in subset])
        y_c = np.array([y for _, y in subset])
        clients.append((X_c, y_c))

    return clients

# ===============================
# 4. Poisoning Attack
# ===============================
def poison_data(X, y):
    y_poison = y.copy()
    y_poison = 1 - y_poison  # label flipping
    return X, y_poison

# ===============================
# 5. Robust Aggregation (Core)
# ===============================
def robust_aggregation(weights_list):
    weights_array = np.array(weights_list)

    # Compute mean
    mean_weights = np.mean(weights_array, axis=0)

    # Compute distance from mean
    distances = np.linalg.norm(weights_array - mean_weights, axis=1)

    # Remove outliers (top 20%)
    threshold = np.percentile(distances, 80)
    filtered = weights_array[distances < threshold]

    return np.mean(filtered, axis=0)

# ===============================
# 6. Federated Training
# ===============================
def federated_training(X_train, y_train, X_test, y_test):

    NUM_CLIENTS = 10
    ROUNDS = 20
    POISON_RATIO = 0.3

    clients = create_clients(X_train, y_train, NUM_CLIENTS)

    global_model = create_model(X_train.shape[1])
    global_weights = global_model.get_weights()

    for round in range(ROUNDS):
        local_weights = []

        for i, (X_c, y_c) in enumerate(clients):

            model = create_model(X_train.shape[1])
            model.set_weights(global_weights)

            # Poison some clients
            if random.random() < POISON_RATIO:
                X_c, y_c = poison_data(X_c, y_c)

            model.fit(X_c, y_c, epochs=2, batch_size=32, verbose=0)

            weights = model.get_weights()
            flat_weights = np.concatenate([w.flatten() for w in weights])

            local_weights.append(flat_weights)

        # Robust aggregation
        aggregated = robust_aggregation(local_weights)

        # Rebuild weights
        new_weights = []
        idx = 0
        for w in global_weights:
            shape = w.shape
            size = np.prod(shape)
            new_w = aggregated[idx:idx+size].reshape(shape)
            new_weights.append(new_w)
            idx += size

        global_weights = new_weights
        global_model.set_weights(global_weights)

        # Evaluate
        preds = (global_model.predict(X_test) > 0.5).astype(int)
        acc = accuracy_score(y_test, preds)

        print(f"Round {round+1} Accuracy: {acc:.4f}")

    return global_model

# ===============================
# 7. Run Experiment
# ===============================
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data("dataset.csv")

    model = federated_training(X_train, y_train, X_test, y_test)
