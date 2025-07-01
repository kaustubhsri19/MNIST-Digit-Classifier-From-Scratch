# MNIST Digit Classifier from Scratch (NumPy)

Awesome — let's dive into **Neural Network Basics** by building a digit classifier on **MNIST** from scratch using **pure NumPy**.

---

## ✅ Project Overview

We'll build a **3-layer Neural Network**:

* Input Layer → 784 nodes (28x28 images)
* Hidden Layer → 64 nodes (you can tune this)
* Output Layer → 10 nodes (for digits 0–9)

### 🔧 Tasks Breakdown

1. Load MNIST Data
2. Initialize weights and biases
3. Implement:
   * Forward pass (with ReLU & Softmax)
   * Loss function (cross-entropy)
   * Backward pass (gradients)
   * Update weights (using SGD)
4. Train and evaluate

---

## 🧱 Step 1: Load and Preprocess MNIST Data

We'll use `sklearn.datasets.fetch_openml` to load MNIST.

```python
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load MNIST from OpenML
mnist = fetch_openml('mnist_784', version=1)
X = mnist['data'].astype(np.float32) / 255.0  # Normalize [0, 1]
y = mnist['target'].astype(int).reshape(-1, 1)  # Labels 0–9

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False, categories='auto')
y_encoded = encoder.fit_transform(y)

# Split into train/val
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
```

---

## 🧠 Step 2: Define Network Architecture and Initialization

```python
# Network architecture
input_size = 784
hidden_size = 64
output_size = 10

# Initialize weights and biases
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))
```

---

## 🔄 Step 3: Activation Functions & Forward Pass

```python
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def cross_entropy(preds, labels):
    return -np.mean(np.sum(labels * np.log(preds + 1e-8), axis=1))
```

---

## 🔁 Step 4: Training Loop with Backpropagation

```python
def train(X, y, epochs=100, lr=0.1):
    global W1, b1, W2, b2
    for epoch in range(1, epochs + 1):
        # Forward
        Z1 = X @ W1 + b1
        A1 = relu(Z1)
        Z2 = A1 @ W2 + b2
        A2 = softmax(Z2)

        # Loss
        loss = cross_entropy(A2, y)

        # Backward
        dZ2 = A2 - y
        dW2 = A1.T @ dZ2 / X.shape[0]
        db2 = np.mean(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = X.T @ dZ1 / X.shape[0]
        db1 = np.mean(dZ1, axis=0, keepdims=True)

        # Update weights
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        # Accuracy
        if epoch % 10 == 0 or epoch == 1:
            acc = evaluate(X_val, y_val)
            print(f"Epoch {epoch}: Loss = {loss:.4f}, Val Accuracy = {acc:.4f}")

def evaluate(X, y):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)
    preds = np.argmax(A2, axis=1)
    labels = np.argmax(y, axis=1)
    return np.mean(preds == labels)
```

---

## ✅ Step 5: Train the Model

```python
train(X_train, y_train, epochs=100, lr=0.1)
```

---

## 🧪 Done? Test Accuracy

```python
print(f"Final Accuracy: {evaluate(X_val, y_val):.4f}")
```

---

## 🧠 What You Just Learned:

| Concept              | How You Applied It               |
| -------------------- | -------------------------------- |
| Feedforward NN       | Manually coded layer outputs     |
| ReLU & Softmax       | As activation functions          |
| Cross-Entropy Loss   | For multi-class classification   |
| Backpropagation      | Gradient formulas coded manually |
| Weight updates (SGD) | Used gradients to update weights |
