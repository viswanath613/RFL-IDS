from src.models.cnn import build_model
from src.attacks.poisoning import label_flipping, add_noise
import random

def train_client(global_weights, X, y, input_dim, poison=False):
    model = build_model(input_dim)
    model.set_weights(global_weights)

    if poison:
        y = label_flipping(y)

    model.fit(X, y, epochs=2, batch_size=32, verbose=0)

    weights = model.get_weights()

    if poison:
        weights = add_noise(weights)

    return weights
