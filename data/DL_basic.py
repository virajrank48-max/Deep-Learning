import numpy as np

# customer information
age = 30
income = 50 

# weight 
w1 = 0.3
w2 = 0.7
bias = -30

# calculate weight sum
z = (age * w1) + (income * w2) + bias
print("Weighted Sum:", z) 

# apply sigmoid
def segmoid(z):
    return 1/ (1 + np.exp(-z))

prediction = segmoid(z)

print("Prediction:", prediction)

# Building the data: Age and Monthly Income
X = np.array([
    [22, 20],
    [25, 25],
    [35, 60],
    [45, 80],
    [30, 40],
    [40, 70]
], dtype=float)

# target variable 
y = np.array([
     [0],
    [0],
    [1],
    [1],
    [0],
    [1]
], dtype=float)

print("\n")
print("Input shape", X.shape)
print("Target shape", y.shape)

# Normalising the data
X_min = X.min(axis=0)
X_max = X.max(axis = 0)

X_norm = (X - X_min)/(X_max-X_min)
print(X_norm)

np.random.seed(42)

# 2 input features -> 3 hidden neurons
W1 = np.random.randn(2, 3)* 0.1
b1 = np.zeros((1,3))

# 3 hidden neuron -> 1 output neuron
W2 = np.random.randn(3, 1)* 0.1
b2 = np.zeros((1,1))
print("\n")
print("W1 shape:",W1.shape)
print("W2 shape:",W2.shape)

# ReLU (Rectified Linear Unit)
print("""
We will use ReLU (Rectified Linear Unit) for the hidden pattern
and Sigmoid for the output layer""")
print(" ")
print("""ReLU(z) = max(0, z)""")
print("converts negative values into zero and keeps positive values")

print("Sigmoid: sigma(z) = 1 / (1 + e^(-z))")

# convert final output to zero and one
def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

# Hidden layer 
z1 = np.dot(X_norm, W1) + b1
a1 = relu(z1)

# output layer
z2 = np.dot(a1, W2) + b2
a2 = sigmoid(z2)

print("Predictable Probabilities:")
print(a2)

# Loss Function
print("Loss Function - Binary Cross-Entropy (BCE):")
print("L = -1/n * sum(y_i * log(p_i) + (1 - y_i) * log(1 - p_i))")

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-9
    y_pred = np.clip(y_pred, epsilon, 1-epsilon)

    loss = -np.mean(
        y_true*np.log(y_pred) + (1 - y_true) *np.log(1- y_pred)
    )

    return loss

loss = binary_cross_entropy(y, a2)
print("Initial Loss: ", loss)

# Creating Dataset Again

import numpy as np

X = np.array([
    [22, 20],
    [25, 25],
    [35, 60],
    [45, 80],
    [30, 40],
    [40, 70]
], dtype=float)

y = np.array([
    [0],
    [0],
    [1],
    [1],
    [0],
    [1]
], dtype=float)

print(X.shape)
print(y.shape)

print("""Backpropagation is the process of calculating how much each weight contributes to the prediction error, " \
"working backwards from the output layer towards the input layer.""")

print("Gradiant Decent: W_new = W_old - alpha * (dL / dW)")
print("""Where:
W = weight
L = loss function
alpha = learning rate
(dL / dW) = Gradient""")

# prepare the data
X_min = X.min(axis=0)
X_max = X.max(axis=0)

X_norm = (X - X_min) / (X_max - X_min)

# Initialize weight
np.random.seed(42)

W1 = np.random.randn(2, 3) * 0.1
b1 = np.zeros((1, 3))

W2 = np.random.randn(3, 1) * 0.1
b2 = np.zeros((1, 1))

print("W1:", W1.shape)
print("b1:", b1.shape)
print("W2:", W2.shape)
print("b2:", b2.shape)
# Define Activation function and loss
def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def binary_cross_entropy(y_true, y_pred):
    eps = 1e-9
    y_pred = np.clip(y_pred, eps, 1 - eps)

    return -np.mean(
        y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    )


# Backpropogation
print("Backpropogation Steps")
print("dZ2 = (A2 - y) / m")
print("Then Calculate Gradients:")
print("dW2 = A1.T @ dZ2")
print("db2 = sum(dZ2, axis=1, keepdims=True)")
print("For the hidden layers")
print("dA1 = dZ2 @ W2.T")
print("dZ1 = dA1 * relu_derivative(Z1)")
print("dW1 = X.T @ dZ1")

print("""Where, m is numbrt of observations @(o dot) means element wise multiplication""")

learning_rate = 0.1
epochs = 5000

m = X_norm.shape[0]

for epoch in range(epochs):

    # FORWARD PROPAGATION
    z1 = np.dot(X_norm, W1) + b1
    a1 = relu(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    # LOSS
    loss = binary_cross_entropy(y, a2)

    # OUTPUT LAYER GRADIENTS
    dz2 = (a2 - y) / m

    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)

    # HIDDEN LAYER GRADIENTS
    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * (z1 > 0)

    dW1 = np.dot(X_norm.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # UPDATE PARAMETERS
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")


print("a1:", a1.shape)
print("a2:", a2.shape)
print("dz2:", dz2.shape)
print("W2:", W2.shape)
print("dW2:", dW2.shape)

# Final forward propagation
z1 = np.dot(X_norm, W1) + b1
a1 = relu(z1)

z2 = np.dot(a1, W2) + b2
predictions = sigmoid(z2)

print("Predicted probabilities:")
print(predictions)

print("\nPredicted classes:")
print((predictions >= 0.5).astype(int))

print("\nActual classes:")
print(y)
