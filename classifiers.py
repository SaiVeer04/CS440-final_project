import numpy as np


class Perceptron:
    def __init__(self, input_size, num_classes):
        self.weights = np.random.randn(input_size, num_classes) * 0.01
        self.bias = np.zeros((1, num_classes))

    def forward(self, x):
        return np.dot(x, self.weights) + self.bias

    def train(self, X, y, lr=0.01, epochs=50):
        for epoch in range(epochs):
            for i in range(X.shape[0]):
                # Forward
                scores = self.forward(X[i:i + 1])

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


class CustomNN:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):

        self.W1 = np.random.randn(input_size, hidden_size1) * 0.01
        self.b1 = np.zeros((1, hidden_size1))

        self.W2 = np.random.randn(hidden_size1, hidden_size2) * 0.01
        self.b2 = np.zeros((1, hidden_size2))

        self.W3 = np.random.randn(hidden_size2, output_size) * 0.01
        self.b3 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.relu(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)

        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.softmax(self.z3)

        return self.a3

    def backward(self, x, y, lr):
        m = x.shape[0]

        dz3 = self.a3.copy()
        dz3[range(m), y] -= 1
        dz3 /= m

        dW3 = np.dot(self.a2.T, dz3)
        db3 = np.sum(dz3, axis=0, keepdims=True)

        dz2 = np.dot(dz3, self.W3.T) * self.relu_derivative(self.z2)
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        dz1 = np.dot(dz2, self.W2.T) * self.relu_derivative(self.z1)
        dW1 = np.dot(x.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.W3 -= lr * dW3
        self.b3 -= lr * db3
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def train(self, X, y, lr=0.01, epochs=100, batch_size=32):
        for epoch in range(epochs):

            idx = np.arange(X.shape[0])
            np.random.shuffle(idx)

            for i in range(0, X.shape[0], batch_size):
                batch_idx = idx[i:i + batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]

                self.forward(X_batch)

                self.backward(X_batch, y_batch, lr)

    def predict(self, X):
        scores = self.forward(X)
        return np.argmax(scores, axis=1)

    def evaluate(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)
