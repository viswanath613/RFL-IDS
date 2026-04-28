import numpy as np
from src.federated.aggregation import flatten_weights, reshape_weights, robust_aggregation

def aggregate(global_weights, client_weights):
    flat_weights = [flatten_weights(w) for w in client_weights]

    aggregated_flat = robust_aggregation(flat_weights)

    new_weights = reshape_weights(aggregated_flat, global_weights)
    return new_weights
