import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def plot_weight_distributions(model, title=None):
    """Plot weight distributions of model."""
    layers_with_weights = [
        (name, module)
        for name, module in model.named_modules()
        if hasattr(module, "weight") and not isinstance(module, nn.CrossEntropyLoss)
    ]

    fig, axs = plt.subplots(
        len(layers_with_weights), 1, figsize=(10, len(layers_with_weights) * 5)
    )
    if len(layers_with_weights) == 1:
        axs = [axs]

    for i, (name, layer) in enumerate(layers_with_weights):
        weights = layer.weight.data.cpu().numpy()
        weights = weights.flatten()
        ax = axs[i]
        sns.histplot(weights, kde=True, ax=ax, bins=30)
        ax.set_title(f"Layer {i+1}: {name} Weight Distribution")

    plt.suptitle(f"{title}", fontsize=24)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # To provide space for the global title
    plt.show()


def plot_activation_maps(model, dataloader, num_images=2):
    """Plots activation maps for each filter in each convolutional layer of model."""

    # Get random batch of images from the dataloader
    data_iter = iter(dataloader)
    images, _ = next(data_iter)

    # Select random subset of images
    indices = torch.randint(0, len(images), (num_images,))
    images = images[indices]

    # Get convolutional layers of model
    conv_layers = [
        module for module in model.modules() if isinstance(module, nn.Conv2d)
    ]

    # Move model to device of images
    device = images.device
    model.to(device)

    # Plot activation maps for each selected image
    for i, image in enumerate(images):
        plt.figure()

        # If the image has more than 1 channel, convert it to grayscale for displaying
        if image.shape[0] > 1:
            image_gray = image.mean(dim=0)
        else:
            image_gray = image[0]

        plt.imshow(image_gray, cmap="gray")
        plt.title(f"Input Image {i + 1}")
        plt.show()

        x = image.unsqueeze(0)  # Add batch dimension

        # Iterate over each convolutional layer
        for j, layer in enumerate(conv_layers):
            x = layer(x)
            x = F.relu(x)  # Apply ReLU activation

            # Plot all activation maps for the current layer
            num_filters = x.shape[1]
            num_cols = int(np.sqrt(num_filters))
            num_rows = num_filters // num_cols + int(num_filters % num_cols != 0)

            fig, axs = plt.subplots(
                num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3)
            )

            for k in range(num_filters):
                row = k // num_cols
                col = k % num_cols
                if num_rows == 1:  # If only 1 row, axs is a 1D array
                    ax = axs[col]
                else:
                    ax = axs[row, col]

                ax.imshow(x[0][k].detach().cpu().numpy(), cmap="gray")
                ax.set_title(f"Filter {k + 1}")
                ax.axis("off")

            fig.suptitle(f"Conv Layer {j + 1} Activation Maps for Image {i + 1}")
            plt.tight_layout()
            plt.show()


def plot_predicted_probabilities(model, data_loader, num_batches=10):
    """
    Compute and plot the model's maximum predicted probability for a number of batches
    from the provided data loader.

    Args:
        model: Trained model.
        data_loader: DataLoader for the data you wish to visualize.
        num_batches: Number of batches to process.
    """
    max_probs = []

    # Ensure we're in evaluation mode
    model.eval()

    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader):
            if i >= num_batches:
                break

            # Compute predicted probabilities
            x = data.to(next(model.parameters()).device)
            y_pred = model(x)

            # Find maximum probability for each sample and store
            max_prob, _ = torch.max(y_pred, dim=1)
            max_probs.extend(max_prob.cpu().numpy())

    # Plot histogram of maximum predicted probabilities
    plt.figure(figsize=(10, 5))
    sns.histplot(max_probs, bins=30, kde=False)
    plt.xlabel("Maximum predicted probability")
    plt.ylabel("Count")
    plt.title("Histogram of maximum predicted probabilities")
    plt.grid(True)
    plt.show()
