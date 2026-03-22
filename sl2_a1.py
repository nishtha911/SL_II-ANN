import matplotlib.pyplot as plt
import math

# --- Implementation from Scratch ---

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

def relu(x):
    return max(0, x)

def leaky_relu(x, alpha=0.1):
    return x if x > 0 else alpha * x

# --- Data Preparation ---

# Generating a range of values from -10 to 10
x_values = [x * 0.1 for x in range(-100, 101)]

y_sigmoid = [sigmoid(x) for x in x_values]
y_tanh = [tanh(x) for x in x_values]
y_relu = [relu(x) for x in x_values]
y_leaky = [leaky_relu(x) for x in x_values]

# --- Plotting ---

plt.figure(figsize=(12, 8))

# Sigmoid
plt.subplot(2, 2, 1)
plt.plot(x_values, y_sigmoid, color='blue')
plt.title("Sigmoid")
plt.grid(True)

# Tanh
plt.subplot(2, 2, 2)
plt.plot(x_values, y_tanh, color='red')
plt.title("Tanh")
plt.grid(True)

# ReLU
plt.subplot(2, 2, 3)
plt.plot(x_values, y_relu, color='green')
plt.title("ReLU")
plt.grid(True)

# Leaky ReLU
plt.subplot(2, 2, 4)
plt.plot(x_values, y_leaky, color='orange')
plt.title("Leaky ReLU")
plt.grid(True)

plt.tight_layout()
plt.show()