import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision import transforms

from tools import register_hooks


def plot_results(
    models,
    model_name,
    title=None,
):
    """
    Function to plot the results after training a model.

    Parameters
    ----------
    models : dict
        Dictionary containing instances of ModelInfo class.
    model_name : str
        The key of the model's instance in the models dictionary to be plotted.
    title : str, optional
        The title of the plot. Default is None.
    """

    epochs = models[f"{model_name}"].epochs
    losses = models[f"{model_name}"].losses
    train_accuracies = models[f"{model_name}"].train_accuracies
    test_accuracies = models[f"{model_name}"].test_accuracies

    # Initialize figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot losses on ax1
    ax1.plot(epochs, losses)
    ax1.set_xlabel("Epoch Number")
    ax1.set_ylabel("Cross Entropy")
    ax1.set_title("Cross Entropy")

    # Plot accuracies on ax2
    ax2.plot(range(len(train_accuracies)), train_accuracies, label="Training Accuracy")
    ax2.plot(range(len(test_accuracies)), test_accuracies, label="Test Accuracy")
    ax2.set_xlabel("Epoch number")
    ax2.set_ylabel("Accuracy, in %")
    ax2.set_title("Accuracy")

    plt.legend()
    plt.suptitle(f"{title}", fontsize=28)
    plt.show()


def plot_reg_results(
    models,
    model_name,
    title=None,
):
    """
    Function to plot the results after training models with regularization.

    Parameters
    ----------
    models : dict
        Dictionary containing instances of ModelInfo class.
    model_name : str
        The key of the model's instance in the models dictionary to be plotted.
    title : str, optional
        The title of the plot. Default is None.
    """

    epochs = models[f"{model_name}"].epochs
    losses = models[f"{model_name}"].losses
    reg_losses = models[f"{model_name}"].reg_losses
    train_accuracies = models[f"{model_name}"].train_accuracies
    test_accuracies = models[f"{model_name}"].test_accuracies

    # Initialize figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot losses on ax1
    ax1.plot(epochs, losses, label="Total Loss")
    ax1.plot(epochs, reg_losses, label="Regularization Loss")
    ax1.plot(
        epochs,
        np.asarray(losses) - np.asarray(reg_losses),
        label="Cross Entropy Loss",
    )
    ax1.set_xlabel("Epoch Number")
    ax1.set_ylabel("Loss")
    ax1.set_title("Losses")

    # Plot accuracies on ax2
    ax2.plot(range(len(train_accuracies)), train_accuracies, label="Training Accuracy")
    ax2.plot(range(len(test_accuracies)), test_accuracies, label="Test Accuracy")
    ax2.set_xlabel("Epoch number")
    ax2.set_ylabel("Accuracy, in %")
    ax2.set_title("Accuracy")

    ax1.legend()
    ax2.legend()
    plt.suptitle(f"{title}", fontsize=28)
    plt.show()


def plot_weight_distributions(model, title=None):
    """Plot weight distributions of the given model.

    This function goes through each layer of the model, extracts the weights
    if the layer has any, and plots a histogram of the weight values.

    Args:
        model (nn.Module): The model whose weights to plot.
        title (str, optional): The title for the plot. Defaults to None, in which case no title is displayed.

    """

    # Create a list of all the layers in the model that have weights.
    # Note: We exclude layers of type nn.CrossEntropyLoss because these don't have weights.
    layers_with_weights = [
        (name, module)
        for name, module in model.named_modules()
        if hasattr(module, "weight") and not isinstance(module, nn.CrossEntropyLoss)
    ]

    # Initialize a matplotlib figure with one subplot per layer with weights
    fig, axs = plt.subplots(
        len(layers_with_weights), 1, figsize=(10, len(layers_with_weights) * 5)
    )

    # If there's only one layer with weights, axs would be just one AxesSubplot object.
    # In that case, make it a single-item list so that we can iterate over axs in the following for loop.
    if len(layers_with_weights) == 1:
        axs = [axs]

    # Loop over the layers with weights
    for i, (name, layer) in enumerate(layers_with_weights):
        # Get the weights of the layer and flatten the array to 1D for histogram plotting
        weights = layer.weight.data.cpu().numpy()
        weights = weights.flatten()

        # Plot a histogram of the weights in the current layer
        ax = axs[i]
        sns.histplot(weights, kde=True, ax=ax, bins=30)
        ax.set_title(f"Layer {i+1}: {name} Weight Distribution")

    plt.suptitle(f"{title}", fontsize=24)

    # Adjust the layout so that the subplots don't overlap and there's space for the suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_activation_maps(model, dataloader, num_images=1, dataset="mnist"):
    """Plots activation maps for each filter in each convolutional layer of the model.

    Args:
        model (nn.Module): The model whose activations to plot.
        dataloader (DataLoader): The DataLoader to use for obtaining input images.
        num_images (int, optional): Number of images to plot activations for. Defaults to 1.
        dataset (str, optional): Name of the dataset ("mnist", "cifar10", or "cifar100"). Defaults to "mnist".

    Raises:
        ValueError: If the dataset name is not one of the expected values.
    """

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

    # Move model to the same device as the images
    device = images.device
    model.to(device)

    # Use a grayscale colormap if the dataset is MNIST, otherwise use the default colormap
    cmap = "gray" if dataset == "mnist" else None

    # Plot activation maps for each selected image
    for i, image in enumerate(images):
        plt.figure()

        # Un-normalize the image:
        if dataset == "mnist":
            image = image * 0.3081 + 0.1307
            image_gray = image[0]  # MNIST only has one channel
        elif dataset == "cifar10":
            image = image * 0.5 + 0.5
            image_gray = image.permute(1, 2, 0)  # Move channels to the end for display
        elif dataset == "cifar100":
            mean = torch.tensor([0.5071, 0.4865, 0.4409]).unsqueeze(-1).unsqueeze(-1)
            std = torch.tensor([0.2009, 0.1984, 0.2023]).unsqueeze(-1).unsqueeze(-1)
            image = std * image + mean
            image_gray = image.permute(1, 2, 0)  # Move channels to the end for display
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        plt.imshow(image_gray)  # Plot original image in original colours
        plt.title(f"Input Image {i + 1}")
        plt.show()

        x = image.unsqueeze(0)  # Add batch dimension

        # Iterate over each convolutional layer
        for j, layer in enumerate(conv_layers):
            x = layer(x)  # Pass the image through the layer to get its activations.

            # Apply activation function based on dataset
            if dataset == "mnist":
                x = torch.tanh(x)  # LeNet activation function
            elif dataset == "cifar10" or dataset == "cifar100":
                x = F.relu(x)  # DDNet activation function

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

                ax.imshow(x[0][k].detach().cpu().numpy(), cmap=cmap)
                ax.set_title(f"Filter {k + 1}")
                ax.axis("off")

            fig.suptitle(
                f"Conv Layer {j + 1} Activation Maps for Image {i + 1}",
                fontsize="large",
            )
            plt.tight_layout()
            plt.show()


def plot_max_predicted_scores(model, data_loader, num_batches=10):
    """
    Compute and plot the model's maximum predicted scores (often slightly faultily interpreted as probability) for a number of batches
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

            # Compute predicted scores
            x = data.to(next(model.parameters()).device)
            y_pred = model(x)

            # Find maximum predicted scores for each sample and store
            max_prob, _ = torch.max(y_pred, dim=1)
            max_probs.extend(max_prob.cpu().numpy())

    # Plot histogram of maximum predicted scores
    plt.figure(figsize=(10, 5))
    sns.histplot(max_probs, bins=30, kde=False)
    plt.xlabel("Maximum predicted score")
    plt.ylabel("Count")
    plt.title("Histogram of maximum predicted scores")
    plt.grid(True)
    plt.show()


def plot_activations_pca(model, data_loader, device, dataset="mnist"):
    """Plots PCA of activations for each layer of the model.

    Args:
        model (nn.Module): The model whose activations to plot.
        data_loader (DataLoader): The DataLoader to use for obtaining input images.
        device (str): The device to move the data to (e.g., "cuda" or "cpu").
        dataset (str, optional): Name of the dataset ("mnist", "cifar10", or "cifar100"). Defaults to "mnist".
    """
    num_classes = (
        100 if dataset == "cifar100" else 10
    )  # Number of classes depends on dataset
    colors = cm.get_cmap("rainbow", num_classes)(
        np.arange(num_classes)
    )  # Generate colors for each class

    # Register hooks on model layers to save output
    save_output, hook_handles, layer_names = register_hooks(model)
    model.eval()

    with torch.no_grad():  # No gradient computation required for PCA
        for images, batch_labels in data_loader:  # Load batches of images and labels
            images = images.to(device)
            model(images)  # Forward pass
            batch_labels = batch_labels.cpu().numpy()
            break  # Only process one batch

        # For each layer's output
        for i, output in enumerate(save_output.outputs):
            output = (
                output.view(output.size(0), -1).cpu().numpy()
            )  # Flatten output tensor, excluding the batch dimension

            # Compute PCA and transform the output
            pca = PCA(n_components=2)
            result = pca.fit_transform(output)

            # Plot the transformed output
            plt.figure(figsize=(6, 6))
            added_labels = set()
            for j in range(len(result)):  # For each datapoint
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

            # Sorting labels and handles in order of class labels
            handles, legend_labels = plt.gca().get_legend_handles_labels()
            by_label = {label: handle for label, handle in zip(legend_labels, handles)}
            ordered_labels = sorted(by_label.keys(), key=int)
            ordered_handles = [by_label[label] for label in ordered_labels]
            plt.legend(ordered_handles, ordered_labels)

            plt.title(f"PCA of {layer_names[i]} Layer")
            plt.show()

    for handle in hook_handles:  # Remove hooks after use
        handle.remove()

    save_output.clear()


def plot_activations_tsne(model, data_loader, device, dataset="mnist"):
    """Plots t-SNE of activations for each layer of the model.

    Args:
        model (nn.Module): The model whose activations to plot.
        data_loader (DataLoader): The DataLoader to use for obtaining input images.
        device (str): The device to move the data to (e.g., "cuda" or "cpu").
        dataset (str, optional): Name of the dataset ("mnist", "cifar10", or "cifar100"). Defaults to "mnist".
    """
    # Number of classes is specific to dataset
    num_classes = 100 if dataset == "cifar100" else 10
    colors = cm.get_cmap("rainbow", num_classes)(np.arange(num_classes))

    # Register hooks on model layers to save output
    save_output, hook_handles, layer_names = register_hooks(model)
    model.eval()

    # No gradient computation required for t-SNE
    with torch.no_grad():
        # Load a batch of images and labels
        for images, batch_labels in data_loader:
            images = images.to(device)
            # Forward pass through the model
            model(images)
            # Converting labels to numpy array
            batch_labels = batch_labels.cpu().numpy()
            # Only process one batch
            break

        # For each layer's output
        for i, output in enumerate(save_output.outputs):
            # Flatten the output tensor, excluding the batch dimension
            output = output.view(output.size(0), -1).cpu().numpy()

            # Compute t-SNE and transform the output
            tsne = TSNE(n_components=2, random_state=0)
            result = tsne.fit_transform(output)

            # Plotting the transformed output
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

            # Sorting the labels and handles in order of class labels
            handles, legend_labels = plt.gca().get_legend_handles_labels()
            by_label = {label: handle for label, handle in zip(legend_labels, handles)}
            ordered_labels = sorted(by_label.keys(), key=int)
            ordered_handles = [by_label[label] for label in ordered_labels]
            plt.legend(ordered_handles, ordered_labels)

            plt.title(f"t-SNE of {layer_names[i]} Layer")
            plt.show()

    # Remove the hooks after use
    for handle in hook_handles:
        handle.remove()

    save_output.clear()


def plot_saliency_maps(model, data_loader, num_images, dataset="mnist"):
    """Plots saliency maps of the first num_images images in data_loader.

    Args:
        model (nn.Module): The model for which to calculate saliency maps.
        data_loader (DataLoader): The DataLoader to use for obtaining input images.
        num_images (int): The number of images for which to plot saliency maps.
        dataset (str, optional): Name of the dataset ("mnist", "cifar10", or "cifar100"). Defaults to "mnist".
    """
    # Set the model to evaluation mode
    model.eval()

    count = 0

    # Create a grid to plot the images and saliency maps
    rows = int(np.ceil(np.sqrt(num_images)))
    cols = 3
    fig, axs = plt.subplots(rows, cols, figsize=(12, 10))

    for images, labels in data_loader:
        # Enable gradient calculation for the images
        images.requires_grad = True

        # Forward pass
        outputs = model(images)

        # Calculate gradients of the output with respect to the input images
        model.zero_grad()
        batch_size = images.size(0)
        for i in range(batch_size):
            if count >= num_images:
                break

            # Calculate gradient of output with respect to images
            outputs[i, labels[i]].backward(retain_graph=True)

            # Get the gradients for the i-th image
            gradients = images.grad.data[i]

            # Convert the gradients to grayscale
            grayscale_gradients = torch.mean(gradients, dim=0, keepdim=True)

            # Normalize the gradients between 0 and 1
            normalized_gradients = torch.abs(grayscale_gradients).squeeze()
            normalized_gradients = (
                normalized_gradients - normalized_gradients.min()
            ) / (normalized_gradients.max() - normalized_gradients.min())

            # Convert the normalized gradients to a numpy array
            saliency_map = normalized_gradients.detach().cpu().numpy()

            # Unnormalize the image data depending on the dataset
            img = images[i].detach().cpu().numpy()
            if dataset == "mnist":
                img = img * 0.3081 + 0.1307
            elif dataset == "cifar10":
                img = img * 0.5 + 0.5
            elif dataset == "cifar100":
                mean = np.array([0.5071, 0.4865, 0.4409]).reshape(3, 1, 1)
                std = np.array([0.2009, 0.1984, 0.2023]).reshape(3, 1, 1)
                img = std * img + mean

            # Transpose the image data and plot
            img = np.transpose(img, (1, 2, 0))

            # Plot the image
            ax_img = axs[count // cols, 0]
            ax_img.imshow(img)
            ax_img.axis("off")

            # Plot the saliency map
            ax_saliency = axs[count // cols, 1]
            ax_saliency.imshow(saliency_map, cmap="hot", alpha=0.5)
            ax_saliency.axis("off")

            # Plot the overlapping image
            ax_overlap = axs[count // cols, 2]
            ax_overlap.imshow(img)
            ax_overlap.imshow(saliency_map, cmap="hot", alpha=0.5)
            ax_overlap.axis("off")

            count += 1

        if count >= num_images:
            break

    # Remove any remaining empty subplots
    for i in range(count, rows * cols):
        axs.flatten()[i].remove()

    # Adjust spacing and display the plot
    plt.tight_layout()
    plt.show()


def plot_occlusion_sensitivity(
    model, data_loader, num_images, occluder_size=8, stride=4, dataset="mnist"
):
    """
    Plots occlusion sensitivity heatmaps for images in the provided data_loader.
    The occlusion sensitivity is calculated by sliding an occluder (a patch of zero or mean pixel values)
    across the image and observing the effect on the model's output confidence for the true class.

    Args:
        model (nn.Module): The trained model.
        data_loader (DataLoader): The DataLoader for obtaining input images.
        num_images (int): The number of images for which to plot occlusion sensitivity.
        occluder_size (int, optional): The side length of the square occluder. Defaults to 8.
        stride (int, optional): The stride for the occluder. Defaults to 4.
        dataset (str, optional): The dataset the images come from. This affects the normalization used. Defaults to "mnist".
    """
    # Set model to evaluation mode
    model.eval()

    # Set up subplots for the images and their occlusion sensitivity maps
    fig, axs = plt.subplots(num_images, 2, figsize=(10, num_images * 5))

    # Define the occluder based on dataset
    occluder = (
        torch.zeros(1, occluder_size, occluder_size)
        if dataset == "mnist"
        else torch.ones(3, occluder_size, occluder_size)
    )

    # Define denormalization transforms
    if dataset == "cifar100":
        denorm = transforms.Normalize(
            mean=torch.tensor([-0.5071 / 0.2009, -0.4865 / 0.1984, -0.4409 / 0.2023]),
            std=torch.tensor([1 / 0.2009, 1 / 0.1984, 1 / 0.2023]),
        )
    else:
        denorm = lambda x: x / 2 + 0.5

    count = 0
    for images, _ in data_loader:
        for image in images:
            if count >= num_images:
                break

            # Obtain the dimensions of the input image
            image_height, image_width = image.shape[-2:]

            # Calculate the number of occluders that can fit along each dimension
            num_occluders_height = (image_height - occluder_size) // stride + 1
            num_occluders_width = (image_width - occluder_size) // stride + 1

            # Create a grid to store the occlusion scores
            occlusion_scores = np.zeros((num_occluders_height, num_occluders_width))

            # Forward pass through the model to get the predicted class
            output = model(image.unsqueeze(0))
            predicted_class = torch.argmax(output, dim=1).item()

            # Iterate through all occluders and calculate the model's output confidence
            for i in range(num_occluders_height):
                for j in range(num_occluders_width):
                    # Occlude the region of the image with the occluder
                    occluded_image = image.clone()
                    occluded_image[
                        :,
                        i * stride : i * stride + occluder_size,
                        j * stride : j * stride + occluder_size,
                    ] = occluder

                    # Forward pass through the model
                    output = model(occluded_image.unsqueeze(0))
                    confidence = F.softmax(output, dim=1)[0, predicted_class].item()

                    # Store the confidence score in the occlusion scores grid
                    occlusion_scores[i, j] = confidence

            # Normalize the occlusion scores between 0 and 1
            occlusion_scores = (occlusion_scores - occlusion_scores.min()) / (
                occlusion_scores.max() - occlusion_scores.min() + 1e-8
            )

            # Plot the original image on the left subplot
            if dataset == "mnist":
                axs[count, 0].imshow(image.squeeze(), cmap="gray")
            else:
                # reverse the normalization for CIFAR10/CIFAR100 before plotting
                image_denorm = denorm(image)
                axs[count, 0].imshow(np.transpose(image_denorm.numpy(), (1, 2, 0)))
            axs[count, 0].axis("off")

            # Plot the occlusion scores heatmap on the right subplot
            heatmap = axs[count, 1].imshow(
                occlusion_scores, cmap="hot", interpolation="nearest"
            )
            axs[count, 1].axis("off")

            # Add a colorbar for the heatmap
            fig.colorbar(heatmap, ax=axs[count, 1])

            count += 1
        if count >= num_images:
            break

    # Show the plot
    plt.tight_layout()
    plt.show()
