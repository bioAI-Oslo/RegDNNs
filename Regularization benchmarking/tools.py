import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm


def train(
    train_loader,
    test_loader,
    model,
    n_epochs=2,
    l1=False,
    l1_lmbd=0.00001,
    l2=False,
    l2_lmbd=0.0001,
    l1_l2=False,
    soft_svb=False,
    soft_svb_lmbd=0.01,
    hard_svb=False,
    hard_svb_lmbd=0.001,
    jacobi_reg=False,
    jacobi_reg_lmbd=0.001,
    jacobi_det_reg=False,
    jacobi_det_reg_lmbd=0.001,
    conf_penalty=False,
    conf_penalty_lmbd=0.1,
    label_smoothing=False,
    label_smoothing_lmbd=0.1,
    hessian_reg=False,
    hessian_reg_lmbd=0.001,
):
    losses = []
    epochs = []
    weights = []
    train_accuracies = []
    test_accuracies = []
    reg_losses = []

    for epoch in tqdm(range(n_epochs)):
        N = len(train_loader)
        for param in model.parameters():
            weights.append(param.detach().numpy().copy())
        for i, (data, labels) in enumerate(train_loader):
            epochs.append(epoch + i / N)
            loss_data, reg_loss_data = model.train_step(
                data,
                labels,
                l1=l1,
                l1_lmbd=l1_lmbd,
                l2=l2,
                l2_lmbd=l2_lmbd,
                l1_l2=l1_l2,
                soft_svb=soft_svb,
                soft_svb_lmbd=soft_svb_lmbd,
                jacobi_reg=jacobi_reg,
                jacobi_reg_lmbd=jacobi_reg_lmbd,
                jacobi_det_reg=jacobi_det_reg,
                jacobi_det_reg_lmbd=jacobi_det_reg_lmbd,
                conf_penalty=conf_penalty,
                conf_penalty_lmbd=conf_penalty_lmbd,
                label_smoothing=label_smoothing,
                label_smoothing_lmbd=label_smoothing_lmbd,
                hessian_reg=hessian_reg,
                hessian_reg_lmbd=hessian_reg_lmbd,
            )
            losses.append(loss_data)
            reg_losses.append(reg_loss_data)

        if hard_svb:
            svb(model, eps=hard_svb_lmbd)

        train_accuracies.append(accuracy(model, test_loader))
        test_accuracies.append(accuracy(model, train_loader))
        model.counter = 0
        print(f"Epoch: {epoch}")
        print(
            "Accuracy of the network on the test images: %d %%"
            % (100 * accuracy(model, test_loader))
        )
    return losses, reg_losses, epochs, weights, train_accuracies, test_accuracies


def plot_results(epochs, losses, train_accuracies, test_accuracies, title=None):
    """Plot results after training a model."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    ax1.plot(epochs, losses, "o--")
    ax1.set_xlabel("Epoch Number")
    ax1.set_ylabel("Cross Entropy")
    ax1.set_title("Cross Entropy")

    ax2.plot(
        range(len(train_accuracies)), train_accuracies, "o--", label="Training Accuracy"
    )
    ax2.plot(range(len(test_accuracies)), test_accuracies, "o--", label="Test Accuracy")
    ax2.set_xlabel("Epoch number")
    ax2.set_ylabel("Accuracy, in %")
    ax2.set_title("Accuracy")

    plt.legend()
    plt.suptitle(f"{title}", fontsize=28)
    plt.show()


def plot_reg_results(
    epochs, losses, reg_losses, train_accuracies, test_accuracies, title=None
):
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

    ax2.plot(
        range(len(train_accuracies)), train_accuracies, "o--", label="Training Accuracy"
    )
    ax2.plot(range(len(test_accuracies)), test_accuracies, "o--", label="Test Accuracy")
    ax2.set_xlabel("Epoch number")
    ax2.set_ylabel("Accuracy, in %")
    ax2.set_title("Accuracy")

    ax1.legend()
    ax2.legend()
    plt.suptitle(f"{title}", fontsize=28)
    plt.show()


def accuracy(model, loader):
    """Calculate the accuracy of a model. Uses a data loader."""
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
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
