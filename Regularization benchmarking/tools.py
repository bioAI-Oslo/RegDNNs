import matplotlib.pyplot as plt
import torch
import numpy as np


def plot_results(epochs, losses, accuracies, title=None):
    """Plot results after training a model."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    ax1.plot(epochs, losses, "o--")
    ax1.set_xlabel("Epoch Number")
    ax1.set_ylabel("Cross Entropy")
    ax1.set_title("Cross Entropy")

    ax2.plot(range(len(accuracies)), accuracies, "o--")
    ax2.set_xlabel("Epoch number")
    ax2.set_ylabel("Accuracy, in %")
    ax2.set_title("Accuracy")

    plt.suptitle(f"{title}", fontsize=28)
    plt.show()


def plot_reg_results(epochs, losses, reg_losses, accuracies, title=None):
    """Plot results after training a model with regularization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    ax1.plot(epochs, losses, "o--", label="Total Loss")
    ax1.plot(epochs, reg_losses, "o--", label="Regularization Loss")
    ax1.plot(
        epochs,
        np.asarray(losses) - np.asarray(reg_losses),
        "o--",
        label="Cross Entropy Loss",
    )
    ax1.set_xlabel("Epoch Number")
    ax1.set_ylabel("Loss")
    ax1.set_title("Losses")

    ax2.plot(range(len(accuracies)), accuracies, "o--")
    ax2.set_xlabel("Epoch number")
    ax2.set_ylabel("Accuracy, in %")
    ax2.set_title("Accuracy")

    ax1.legend()
    plt.suptitle(f"{title}", fontsize=28)
    plt.show()


def accuracy(model, test_loader):
    """Calculate the accuracy of a model. Uses a test_loader data loader."""
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def svb(model, eps=0.001):
    """Implements hard singular value bounding as described in Jia et al. 2019.
    Keyword Arguments:
        eps -- Small constant that sets the weights a small interval around 1 (default: {0.001})
    """
    old_weights = model.fc1.weight.data.clone().detach()
    w = torch.linalg.svd(old_weights, full_matrices=False)
    U, sigma, V = w[0], w[1], w[2]
    for i in range(len(sigma)):
        if sigma[i] > 1 + eps:
            sigma[i] = 1 + eps
        elif sigma[i] < 1 / (1 + eps):
            sigma[i] = 1 / (1 + eps)
        else:
            pass
    new_weights = U @ torch.diag(sigma) @ V
    model.fc1.weight.data = new_weights
