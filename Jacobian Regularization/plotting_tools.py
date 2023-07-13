import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA

from tools import register_hooks


def plot_results(
    epochs,
    losses,
    train_accuracies,
    test_accuracies,
    title=None,
):
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
    epochs,
    losses,
    reg_losses,
    train_accuracies,
    test_accuracies,
    title=None,
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


def plot_activations_pca(model, data_loader, device):
    save_output, hook_handles, layer_names = register_hooks(model)
    model.eval()

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    with torch.no_grad():
        for images, batch_labels in data_loader:
            images = images.to(device)
            model(images)
            batch_labels = batch_labels.cpu().numpy()
            break

        for i, output in enumerate(save_output.outputs):
            output = output.view(output.size(0), -1).cpu().numpy()

            pca = PCA(n_components=2)
            result = pca.fit_transform(output)

            plt.figure(figsize=(6, 6))
            added_labels = set()
            for j in range(len(result)):
                label = str(int(batch_labels[j]))
                if label not in added_labels:
                    plt.scatter(
                        result[j, 0],
                        result[j, 1],
                        color=colors[int(label)],
                        label=label,
                    )
                    added_labels.add(label)
                else:
                    plt.scatter(result[j, 0], result[j, 1], color=colors[int(label)])

            handles, legend_labels = plt.gca().get_legend_handles_labels()
            by_label = {label: handle for label, handle in zip(legend_labels, handles)}
            ordered_labels = sorted(by_label.keys(), key=int)
            ordered_handles = [by_label[label] for label in ordered_labels]
            plt.legend(ordered_handles, ordered_labels)

            plt.title(f"PCA of {layer_names[i]} Layer")
            plt.show()

    for handle in hook_handles:
        handle.remove()

    save_output.clear()


def plot_decision_boundary(model, img, device, resolution=100):
    # Make sure the model is in evaluation mode
    model.eval()

    # Flatten the image if necessary and convert to a single precision float
    if img.dim() > 2:
        img = img.view(-1)
    img = img.to(device).float()

    # Generate random vectors
    v1 = torch.randn_like(img)
    v2 = torch.randn_like(img)
    v1 /= v1.norm()
    v2 -= v1 * (v1.dot(v2))  # make orthogonal to v1
    v2 /= v2.norm()

    # Generate grid
    scale = 3  # to define the size of the plane in the image space
    x = torch.linspace(-scale, scale, resolution)
    y = torch.linspace(-scale, scale, resolution)
    xv, yv = torch.meshgrid(x, y)

    # Create the 2D plane passing through the image
    plane = (
        img[None, None, :]
        + xv[..., None] * v1[None, None, :]
        + yv[..., None] * v2[None, None, :]
    )

    # Compute the model prediction
    plane = plane.to(device)
    with torch.no_grad():
        output = model(plane.view(-1, 1, 28, 28)).view(resolution, resolution, -1)
    probs = torch.nn.functional.softmax(output, dim=-1)

    # Draw the figure
    plt.figure(figsize=(8, 8))
    for i in range(10):  # 10 is the number of classes
        # Choose a color map
        cmap = plt.cm.get_cmap("viridis", 10)
        plt.contourf(
            xv.cpu(), yv.cpu(), probs[..., i].cpu(), levels=20, alpha=0.1, cmap=cmap
        )
    plt.show()
