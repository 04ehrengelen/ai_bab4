# Example 4.2 Multiple Layer Perceptron
# Three Inputs
from sklearn.neural_network import MLPClassifier

# Three inputs
X = [
    [0., 0., 0.],
    [0., 0., 1.],
    [0., 1., 0.],
    [0., 1., 1.],
    [1., 0., 0.],
    [1., 0., 1.],
    [1., 1., 0.],
    [1., 1., 1.]
]

# Output for 3-input OR
y = [0, 1, 1, 1, 1, 1, 1, 1]

# Create MLP
clf = MLPClassifier(
    solver='lbfgs',
    alpha=1e-5,
    hidden_layer_sizes=(5, 2),
    random_state=1
)

# Train the model
clf.fit(X, y)

# Test the model
print(clf.predict([
    [0., 0., 0.],
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 1.],
    [1., 1., 1.]
]))

# Show the shape of the weights
print([coef.shape for coef in clf.coefs_])