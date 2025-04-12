import numpy as np

def load_data():
    X_train = np.load("train_data.npy")
    y_train = np.load("train_label.npy")
    X_test = np.load("test_data.npy")
    y_test = np.load("test_label.npy")
    return X_train, y_train, X_test, y_test

def normalize_data(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-8  
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    return X_train_norm, X_test_norm

def get_minibatches(X, y, batch_size=64, shuffle=True):

    N = X.shape[0]
    indices = np.arange(N)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        excerpt = indices[start:end]
        yield X[excerpt], y[excerpt]