import numpy as np
from sklearn.metrics import accuracy_score


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def tanh(values):
    return np.tanh(values)


def binary_cross_entropy(labels, probabilities, epsilon=1e-12):
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    return -np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))


def initialize_parameters(n0, n1, n2, seed=0):
    rng = np.random.default_rng(seed)
    weights_1 = rng.standard_normal((n1, n0)) * np.sqrt(1.0 / n0)
    bias_1 = np.zeros((n1, 1))
    weights_2 = rng.standard_normal((n2, n1)) * np.sqrt(1.0 / n1)
    bias_2 = np.zeros((n2, 1))
    return {
        "W1": weights_1,
        "b1": bias_1,
        "W2": weights_2,
        "b2": bias_2,
    }


def forward_propagation(inputs, parameters):
    weights_1 = parameters["W1"]
    bias_1 = parameters["b1"]
    weights_2 = parameters["W2"]
    bias_2 = parameters["b2"]

    z1 = weights_1 @ inputs + bias_1
    a1 = tanh(z1)
    z2 = weights_2 @ a1 + bias_2
    a2 = sigmoid(z2)

    return {"Z1": z1, "A1": a1, "Z2": z2, "A2": a2}


def backward_propagation(inputs, labels, parameters, activations):
    a1 = activations["A1"]
    a2 = activations["A2"]
    weights_2 = parameters["W2"]

    sample_count = labels.shape[1]

    dz2 = a2 - labels
    dw2 = (dz2 @ a1.T) / sample_count
    db2 = np.sum(dz2, axis=1, keepdims=True) / sample_count

    dz1 = (weights_2.T @ dz2) * (1.0 - np.square(a1))
    dw1 = (dz1 @ inputs.T) / sample_count
    db1 = np.sum(dz1, axis=1, keepdims=True) / sample_count

    return {"dW1": dw1, "db1": db1, "dW2": dw2, "db2": db2}


def update_parameters(parameters, gradients, learning_rate):
    parameters["W1"] = parameters["W1"] - learning_rate * gradients["dW1"]
    parameters["b1"] = parameters["b1"] - learning_rate * gradients["db1"]
    parameters["W2"] = parameters["W2"] - learning_rate * gradients["dW2"]
    parameters["b2"] = parameters["b2"] - learning_rate * gradients["db2"]
    return parameters


def predict(inputs, parameters, threshold=0.5):
    output = forward_propagation(inputs, parameters)["A2"]
    return (output >= threshold).astype(float)


def fit_two_layer_network(inputs, labels, n1=32, learning_rate=0.1, epochs=1000, seed=0):
    n0 = inputs.shape[0]
    n2 = labels.shape[0]

    parameters = initialize_parameters(n0=n0, n1=n1, n2=n2, seed=seed)
    train_loss = []
    train_acc = []

    for _ in range(epochs):
        activations = forward_propagation(inputs, parameters)
        probabilities = activations["A2"]

        loss = binary_cross_entropy(labels, probabilities)
        train_loss.append(loss)

        predictions = (probabilities >= 0.5).astype(float)
        accuracy = accuracy_score(labels.flatten(), predictions.flatten())
        train_acc.append(accuracy)

        gradients = backward_propagation(inputs, labels, parameters, activations)
        parameters = update_parameters(parameters, gradients, learning_rate)

    return {
        "parameters": parameters,
        "loss": train_loss,
        "accuracy": train_acc,
    }
