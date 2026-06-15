import numpy as np
import os
from PIL import Image

class HighAccuracyDigitNN:
    def __init__(self):
        self.w1 = np.random.randn(784, 128) * np.sqrt(2.0 / 784)
        self.b1 = np.zeros((1, 128))
        self.w2 = np.random.randn(128, 64) * np.sqrt(2.0 / 128)
        self.b2 = np.zeros((1, 64))
        self.w3 = np.random.randn(64, 10) * np.sqrt(2.0 / 64)
        self.b3 = np.zeros((1, 10))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(np.float32)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward_propagation(self, inputs, is_training=False):
        self.inputs = np.atleast_2d(inputs)
        
        self.z1 = np.dot(self.inputs, self.w1) + self.b1
        self.a1 = self.relu(self.z1)
        
        if is_training:
            self.dropout_mask1 = (np.random.rand(*self.a1.shape) < 0.8) / 0.8
            self.a1 *= self.dropout_mask1
            
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.relu(self.z2)
        
        if is_training:
            self.dropout_mask2 = (np.random.rand(*self.a2.shape) < 0.8) / 0.8
            self.a2 *= self.dropout_mask2
            
        self.z3 = np.dot(self.a2, self.w3) + self.b3
        self.output = self.softmax(self.z3)
        return self.output

    def train(self, train_inputs, train_outputs, num_train_iterations, initial_lr=0.2):
        m = 1500
        for iteration in range(num_train_iterations):
            learning_rate = initial_lr * (1.0 / (1.0 + 0.001 * iteration))
            
            output = self.forward_propagation(train_inputs, is_training=True)
            error3 = output - train_outputs
            
            error2 = np.dot(error3, self.w3.T)
            delta2 = error2 * self.relu_derivative(self.z2)
            if hasattr(self, 'dropout_mask2'):
                delta2 *= self.dropout_mask2
            
            error1 = np.dot(delta2, self.w2.T)
            delta1 = error1 * self.relu_derivative(self.z1)
            if hasattr(self, 'dropout_mask1'):
                delta1 *= self.dropout_mask1
            
            adj3 = np.dot(self.a2.T, error3) / m
            db3 = np.sum(error3, axis=0, keepdims=True) / m
            
            adj2 = np.dot(self.a1.T, delta2) / m
            db2 = np.sum(delta2, axis=0, keepdims=True) / m
            
            adj1 = np.dot(self.inputs.T, delta1) / m
            db1 = np.sum(delta1, axis=0, keepdims=True) / m
            
            self.w3 -= learning_rate * adj3
            self.b3 -= learning_rate * db3
            self.w2 -= learning_rate * adj2
            self.b2 -= learning_rate * db2
            self.w1 -= learning_rate * adj1
            self.b1 -= learning_rate * db1
            
            if iteration % 100 == 0:
                loss = -np.mean(train_outputs * np.log(output + 1e-8))
                print(f"Iteration {iteration:4d} | Training Loss: {loss:.6f}")

def load_github_csv(filename, max_rows=1500):
    if not os.path.exists(filename):
        print(f"Error: '{filename}' not found.")
        return None, None
    data = np.loadtxt(filename, delimiter=",", skiprows=1, max_rows=max_rows, dtype=np.float32)
    raw_labels = data[:, 0].astype(np.int32)
    X = data[:, 1:] / 255.0
    y = np.zeros((len(raw_labels), 10))
    y[np.arange(len(raw_labels)), raw_labels] = 1.0
    return X, y

if __name__ == "__main__":
    csv_file = "mnist_train.csv"
    X_train, y_train = load_github_csv(csv_file, max_rows=1500)

    if X_train is not None:
        nn = HighAccuracyDigitNN()
        print("Training a high-accuracy model on 1,500 handwriting samples...")
        nn.train(X_train, y_train, num_train_iterations=1500, initial_lr=0.25)

        image_path = "digit.png"
        if os.path.exists(image_path):
            img = Image.open(image_path).convert("L")
            img = img.resize((28, 28))
            test_pattern = np.array(img).flatten() / 255.0
            
            if np.mean(test_pattern) > 0.5:
                test_pattern = 1.0 - test_pattern

            prediction = nn.forward_propagation(test_pattern, is_training=False)
            detected_digit = np.argmax(prediction)

            print(f"\n--- Analysis Complete ---")
            print(f"The updated network looked at '{image_path}' and confidently detected: {detected_digit}")
