# Example 4.1 - Logical OR Neuron

from numpy import exp, array, random, dot

# Set up parameters ===============================================
inputs = array([[0, 0],[0, 1],[1, 0],[1, 1]])

# Target output for Logical OR
outputs = array([[0],[1],[1],[1]])

random.seed(1)

# Initial weights
weights = 2 * random.random((2, 1)) - 1


# Define a Single Neuron function =================================
def neuron(inputs, weights):
    output = 1 / (1 + exp(-(dot(inputs, weights))))
    return output


# Train the Neuron ================================================
for iteration in range(50000):
    output = 1 / (1 + exp(-(dot(inputs, weights))))

    weights += dot(
        inputs.T,
        (outputs - output) * output * (1 - output)
    )


# Test the Neuron =================================================
print("Logical OR Results:")

for x in inputs:
    result = neuron(x, weights)
    print(x, "=>", result)