import math

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(output):
    return output * (1 - output)

# -------- User Inputs --------
x1 = float(input("Enter input x1: "))
x2 = float(input("Enter input x2: "))

w1 = float(input("Enter initial weight w1: "))
w2 = float(input("Enter initial weight w2: "))

b = float(input("Enter initial bias: "))

target = float(input("Enter target output: "))

learning_rate = float(input("Enter learning rate: "))

# -------- Forward Propagation --------
net = x1 * w1 + x2 * w2 + b
output = sigmoid(net)

# -------- Backpropagation --------
error = target - output
delta = error * sigmoid_derivative(output)

# Update weights and bias
w1 = w1 + learning_rate * delta * x1
w2 = w2 + learning_rate * delta * x2
b = b + learning_rate * delta

# -------- Output --------
print("\n----- Results -----")
print(f"Neuron Output          : {output:.4f}")
print(f"Error                  : {error:.4f}")
print(f"Updated Weight w1      : {w1:.4f}")
print(f"Updated Weight w2      : {w2:.4f}")
print(f"Updated Bias           : {b:.4f}")