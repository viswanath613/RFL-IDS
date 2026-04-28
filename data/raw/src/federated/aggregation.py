import numpy as np

def flatten_weights(weights):
    return np.concatenate([w.flatten() for w in weights])

def reshape_weights(flat, template):
    new_weights = []
    idx = 0
    for w in template:
        size = w.size
        new_weights.append(flat[idx:idx+size].reshape(w.shape))
        idx += size
    return new_weights

def robust_aggregation(weights_list):
    weights = np.array(weights_list)
    mean = np.mean(weights, axis=0)
    dist = np.linalg.norm(weights - mean, axis=1)
    threshold = np.percentile(dist, 80)
    filtered = weights[dist < threshold]
    return np.mean(filtered, axis=0)
