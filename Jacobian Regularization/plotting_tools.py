import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import scipy.ndimage as ndi

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


def total_variation(image):
    """Compute the total variation of an image"""
    return np.sum(np.abs(image[:-1, :-1] - image[1:, :-1])) + np.sum(
        np.abs(image[:-1, :-1] - image[:-1, 1:])
    )


def plot_decision_boundary(
    model,
    img,
    v1,
    v2,
    device,
    resolution=300,
    zoom=[0.025, 0.01, 0.001],
    title="No regularization",
):
    # Make sure the model is in evaluation mode
    model.eval()

    # Flatten the image if necessary and convert to a single precision float
    img = img.view(-1)
    if img.dim() > 2:
        img = img.view(-1)
    img = img.to(device).float()

    # Create a figure with n subplots where n is the number of zoom levels
    num_plots = len(zoom)
    fig, axes = plt.subplots(
        1, num_plots, figsize=(8 * num_plots, 8)
    )  # Adjust the figsize as needed

    # If there's only one zoom level, axes will not be a list, so we need to adjust for that
    if num_plots == 1:
        axes = [axes]

    for ax, zoom_level in zip(axes, zoom):
        # Generate grid
        scale = 1 / zoom_level  # to define the size of the plane in the image space
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
        output = model(plane.view(-1, 1, 28, 28)).view(resolution, resolution, -1)
        _, predictions = torch.max(output, dim=-1)

        # with torch.no_grad():
        #     output = model(plane.view(-1, 1, 28, 28)).view(resolution, resolution, -1)
        # probs = torch.nn.functional.softmax(output, dim=-1)

        # Get the class with the highest probability
        # _, predictions = torch.max(probs, dim=-1)

        # Calculate the distance to the closest decision boundary
        decision_boundary = ndi.morphology.distance_transform_edt(
            predictions.cpu().numpy()
            == predictions.cpu().numpy()[resolution // 2, resolution // 2]
        )
        distance_to_boundary = (
            decision_boundary[resolution // 2, resolution // 2] / resolution * 2 * scale
        )

        # Create a colormap where each index i corresponds to the color for digit i
        colors = plt.get_cmap(
            "tab10"
        ).colors  # get the colors used in the 'tab10' colormap
        cmap = ListedColormap([colors[i] for i in range(10)])  # create a new colormap

        # Use 'imshow' instead of 'contourf'; note the 'origin' argument
        img_plot = ax.imshow(
            predictions.cpu(),
            origin="lower",
            extent=(-scale, scale, -scale, scale),
            cmap=cmap,
            alpha=1,
        )

        # Also, let's add the original image as a dot in the middle of our plot
        ax.plot(0, 0, "ro")

        # Draw a circle around the original image with a radius equal to the distance to the closest decision boundary
        circle = plt.Circle((0, 0), distance_to_boundary, color="black", fill=False)
        ax.add_patch(circle)

        # Compute and print the total variation of the decision boundaries
        tv = total_variation(predictions.cpu().numpy())

        # Set title
        ax.set_title(f"Zoom level: {zoom_level}.  Total Variation: {tv}")

    # Set legend for the whole figure
    legend_elements = [
        Patch(facecolor=cmap(i), edgecolor=cmap(i), label=str(i)) for i in range(10)
    ]
    fig.legend(
        handles=legend_elements,
        bbox_to_anchor=(0.97, 1),
        loc="upper left",
        fontsize="large",
        handlelength=2,
    )

    # Add a title to the overall plot
    plt.suptitle(title, fontsize=20)

    plt.show()


def generate_random_vectors(img):
    # Generate random vectors
    img = img.view(-1)
    v1 = torch.randn_like(img)
    v2 = torch.randn_like(img)
    v1 /= v1.norm()
    v2 -= v1 * (v1.dot(v2))  # make orthogonal to v1
    v2 /= v2.norm()
    return v1, v2


def get_random_img(dataloader):
    # Get a batch of data from the dataloader
    images, labels = next(iter(dataloader))
    random_index = np.random.randint(len(labels))
    # Get random image of the batch
    image = images[random_index]

    # If image has a channel dimension (as it should in the case of the MNIST dataset), remove it
    if image.dim() > 2:
        image = image.squeeze(0)

    return image


def plot_and_print_img(image, model, device, regularization_title="no regularization"):
    # Print the prediction of the model
    model.eval()

    with torch.no_grad():
        output = model(image.view(1, 1, 28, 28).to(device))

    # Get the predicted classes
    _, predicted = torch.max(output, 1)

    print(f"Prediction with {regularization_title}: {predicted.item()}")

    # Plot the image
    plt.figure(figsize=(5, 5))
    plt.imshow(image.numpy(), cmap="gray")
    plt.title("Input image")
    plt.show()
