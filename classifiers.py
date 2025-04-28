import numpy as np

class Perceptron:
    def __init__(self, input_size, num_classes):
        self.weights = np.random.randn(input_size, num_classes) * 0.01
        self.bias = np.zeros((1, num_classes))

    def forward(self, x):
        return np.dot(x, self.weights) + self.bias
    
    def train(self, X, y, lr = 0.01, epochs = 100):
        for epoch in range(epochs):
            for i in range(X.shape[0]):
                # Forward
                scores = self.forward(X[i:i+1])

                pred = np.argmax(scores)

                if pred != y[i]:
                    self.weights[:, y[i]] += lr * X[i]
                    self.weights[:, pred] -= lr * X[i]
                    self.bias[0, y[i]] += lr
                    self.bias[0, pred] -= lr
    
    def predict(self, X):
        scores = self.forward(X)

        return np.argmax(scores, axis=1)
    
    def evaluate(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)
        