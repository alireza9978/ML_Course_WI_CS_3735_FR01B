import numpy as np
import pandas as pd


class LinearRegression:

    def __init__(self, learning_rate=0.1, iterations=10000, stopping_threshold=1e-5):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.stopping_threshold = stopping_threshold
        self.m = None
        self.n = None
        self.W = None
        self.b = None
        self.cost_history = None

    def initialize(self):
        self.W = np.random.rand(self.n)
        self.b = np.random.rand()

    def fit(self, x, y):
        if type(x) is pd.DataFrame:
            x = x.values
            y = y.values

        self.m, self.n = x.shape

        # weight initialization
        self.initialize()

        previous_cost = None
        self.cost_history = np.zeros(self.iterations)
        # gradient descent learning
        for i in range(self.iterations):
            y_pred = self.predict(x)
            cost = self.compute_cost(y, y_pred)
            # early stopping criteria
            if previous_cost and abs(previous_cost - cost) <= self.stopping_threshold:
                break
            self.cost_history[i] = cost
            previous_cost = cost
            self.update_weights(x, y)

    def update_weights(self, x, y):
        batch_size = 32
        for batch in range(self.m // batch_size):
            temp_x = x[(batch * batch_size):(batch + 1) * batch_size]
            temp_y = y[(batch * batch_size):(batch + 1) * batch_size]
            y_pred = self.predict(temp_x)
            # calculate gradients
            dW = np.dot(temp_x.T, (y_pred - temp_y)) / batch_size
            db = np.sum(y_pred - temp_y) / batch_size

            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db

    def compute_cost(self, y, y_pred):
        cost = (1 / (2 * self.m)) * np.sum(np.square(y_pred - y))
        return cost

    def score(self, x, y):
        if isinstance(x, pd.DataFrame):
            x = x.values
        if isinstance(y, pd.Series):
            y = y.values
        y_pred = self.predict(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot
        return r_squared

    def predict(self, x):
        return np.dot(x, self.W) + self.b


class LogisticRegression:
    def __init__(self, num_classes, learning_rate=0.01, iterations=10000, stopping_threshold=1e-6):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.stopping_threshold = stopping_threshold
        self.num_classes = num_classes
        self.m = None
        self.n = None
        self.W = None
        self.b = None
        self.cost_history = None

    def compute_cost(self, y_true, y_pred):
        # binary cross entropy
        epsilon = 1e-9
        return -np.sum(y_true * np.log(y_pred + epsilon)) / self.m

    def initialize(self):
        self.W = np.random.rand(self.n, self.num_classes)
        self.b = np.random.rand(self.num_classes)

    def fit(self, x, y):
        if type(x) is pd.DataFrame:
            x = x.values
        if type(y) is pd.DataFrame:
            y = y.values.squeeze()

        self.m, self.n = x.shape

        # convert labels to one-hot encoding
        y_one_hot = np.zeros((self.m, self.num_classes))
        y_one_hot[np.arange(self.m), y] = 1
        y = y_one_hot

        # weight initialization
        self.initialize()

        previous_cost = None
        self.cost_history = np.zeros(self.iterations)
        # gradient descent learning
        for i in range(self.iterations):
            y_pred = self.feed_forward(x)
            cost = self.compute_cost(y, y_pred)
            # early stopping criteria
            if previous_cost and abs(previous_cost - cost) <= self.stopping_threshold:
                break
            self.cost_history[i] = cost
            previous_cost = cost
            self.update_weights(x, y)

    def update_weights(self, x, y):
        batch_size = 32
        for batch in range(self.m // batch_size):
            temp_x = x[(batch * batch_size):(batch + 1) * batch_size]
            temp_y = y[(batch * batch_size):(batch + 1) * batch_size]
            A = self.feed_forward(temp_x)
            dz = A - temp_y  # derivative of sigmoid and bce X.T*(A-y)
            # calculate gradients
            dW = np.dot(temp_x.T, dz) / batch_size
            db = np.sum(dz) / batch_size

            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db

        # Sigmoid method
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1,  keepdims=True)

    def feed_forward(self, x):
        return self.softmax(np.dot(x, self.W) + self.b)

    def predict(self, x):
        y_pred = self.feed_forward(x)
        return np.argmax(y_pred, axis=1)
