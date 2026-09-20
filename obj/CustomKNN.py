import numpy as np

class CustomKNN:

    def __init__(self, k = 5):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        X = np.array(X)
        n = X.shape[0]
        D = np.linalg.norm(X[:, np.newaxis] - self.X_train, axis=2)
        
        predictions = []
        for i in range(n):
            k_indices = np.argsort(D[i])[:self.k]
            k_nearest_label = self.y_train[k_indices]
            predicted_label = np.bincount(k_nearest_label).argmax()
            predictions.append(predicted_label)
            
        return np.array(predictions)