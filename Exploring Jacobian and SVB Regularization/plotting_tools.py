import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import scipy.ndimage as ndi
import torch.nn.functional as F
from tqdm import tqdm

from attack_tools import fgsm_attack_test, pgd_attack_test
from tools import total_variation_isotropic


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
        "o--",
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


def plot_decision_boundary(
    model,
    img,
    v1,
    v2,
    device,
    resolution=250,
    zoom=[0.025, 0.01, 0.001],
    dataset="mnist",
    title="No regularization",
):
    """
    Function to visualize the decision boundaries of a model in a 2D plane.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model.
    img : torch.Tensor
        The input image to construct the 2D plane.
    v1 : torch.Tensor
        First vector to construct the 2D plane.
    v2 : torch.Tensor
        Second vector to construct the 2D plane.
    device : str
        The device to run the model ('cpu' or 'cuda').
    resolution : int, optional
        The resolution of the grid to be used, by default 300.
    zoom : list, optional
        The zoom levels to visualize, by default [0.025, 0.01, 0.001].
    dataset: str
        The dataset the model was trained on.
    title : str, optional
        The title for the plot, by default "No regularization".
    """
    # Enter evaluation mode
    model.eval()

    # Flatten the image if necessary and convert to a single precision float
    img = img.view(-1).to(device).float()

    # Create a figure with subplots for each zoom level
    num_plots = len(zoom)
    fig, axes = plt.subplots(
        1, num_plots, figsize=(8 * num_plots, 8)
    )  # Adjust the figsize as needed

    # Ensure axes is always a list
    if num_plots == 1:
        axes = [axes]

    for ax, zoom_level in zip(axes, zoom):
        # Generate grid of points in plane defined by img, v1 and v2
        scale = 1 / zoom_level  # Scale decided by zoom_level
        x = torch.linspace(-scale, scale, resolution)
        y = torch.linspace(-scale, scale, resolution)
        xv, yv = torch.meshgrid(x, y)

        # Create the 2D plane passing through the image
        plane = (
            img[None, None, :]
            + xv[..., None] * v1[None, None, :]
            + yv[..., None] * v2[None, None, :]
        ).to(device)

        # Compute the model's predictions over the grid
        with torch.no_grad():
            if dataset == "mnist":
                output = model(plane.view(-1, 1, 28, 28)).view(
                    resolution, resolution, -1
                )
            elif dataset == "cifar10" or dataset == "cifar100":
                output = model(plane.view(-1, 3, 32, 32)).view(
                    resolution, resolution, -1
                )

        probs = torch.nn.functional.softmax(output, dim=-1)

        # Get the class with the highest probability
        _, predictions = torch.max(probs, dim=-1)

        # Calculate the distance to the closest decision boundary
        decision_boundary = ndi.morphology.distance_transform_edt(
            predictions.cpu().numpy()
            == predictions.cpu().numpy()[resolution // 2, resolution // 2]
        )
        distance_to_boundary = (
            decision_boundary[resolution // 2, resolution // 2] / resolution * 2 * scale
        )

        # Create a colormap and plot decision boundaries
        if dataset == "cifar100":
            cmap = "nipy_spectral"  # Suitable for many classes
        else:
            colors = plt.get_cmap("tab10").colors
            cmap = ListedColormap([colors[i] for i in range(10)])

        img_plot = ax.imshow(
            predictions.cpu(),
            origin="lower",
            extent=(-scale, scale, -scale, scale),
            cmap=cmap,
            alpha=1,
        )
        ax.plot(0, 0, "ro")  # Plot dot for original image

        # Draw circle around the original image with a radius equal to the distance to the closest decision boundary
        circle = plt.Circle((0, 0), distance_to_boundary, color="black", fill=False)
        ax.add_patch(circle)

        # Compute and print the total variation of the decision boundaries
        tv = total_variation_isotropic(predictions.cpu().numpy())

        ax.set_title(f"Zoom level: {zoom_level}.  Total Variation (isotropic): {tv}")

    if dataset != "cifar100":
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
    """
    Generate two random orthogonal vectors with the same shape as the input image.

    Parameters
    ----------
    img : torch.Tensor
        The input image.

    Returns
    -------
    v1 : torch.Tensor
        The first random vector, normalized.
    v2 : torch.Tensor
        The second random vector, orthogonal to v1 and normalized.
    """
    # Reshape image to 1D tensor
    img = img.view(-1)

    # Generate and normalize random orthogonal vectors
    v1 = torch.randn_like(img)
    v2 = torch.randn_like(img)
    v1 /= v1.norm()
    v2 -= v1 * (v1.dot(v2))  # Make orthogonal
    v2 /= v2.norm()
    return v1, v2


def get_random_img(dataloader):
    """
    Retrieve a random image from a DataLoader.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        The DataLoader to retrieve the image from.

    Returns
    -------
    image : torch.Tensor
        The randomly selected image.
    """
    # Get a batch of data from the dataloader
    images, labels = next(iter(dataloader))

    # Get random image from the batch
    random_index = np.random.randint(len(labels))
    image = images[random_index]

    # If image has a channel dimension, remove it
    if image.dim() > 2:
        image = image.squeeze(0)

    return image


def plot_and_print_img(
    image,
    model,
    device,
    regularization_title="no regularization",
    dataset="mnist",
):
    """
    Plot an image and print the model's prediction for this image.

    Parameters
    ----------
    image : torch.Tensor
        The image to plot and predict.
    model : torch.nn.Module
        The trained model.
    device : str
        The device to run the model ('cpu' or 'cuda').
    dataset: str
        The dataset used to train the model.
    regularization_title : str, optional
        The regularization method used in the model training, by default "no regularization".
    """
    # Ensure model is in evaluation mode
    model.eval()

    # The class names for CIFAR10
    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    # Get and print model's prediction for the image
    with torch.no_grad():
        if dataset == "mnist":
            output = model(image.view(1, 1, 28, 28).to(device))
        elif dataset == "cifar10" or dataset == "cifar100":
            output = model(image.view(1, 3, 32, 32).to(device))

    _, predicted = torch.max(output, 1)

    if dataset == "mnist":
        print(f"Prediction with {regularization_title}: {predicted.item()}")
    elif dataset == "cifar10":
        print(
            f"Prediction with {regularization_title}: {class_names[predicted.item()]}"
        )
    elif dataset == "cifar100":
        pass

    # Plot the image
    plt.figure(figsize=(5, 5))

    # Check if the image is grayscale or color and adjust accordingly
    if dataset == "mnist":  # grayscale
        plt.imshow(image.numpy().squeeze(), cmap="gray")
    elif dataset == "cifar10":  # color
        # Denormalize the image before plotting
        image = image * 0.5 + 0.5
        plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
    elif dataset == "cifar100":  # color
        # Denormalize the image before plotting
        image = (
            image * torch.tensor([0.2009, 0.1984, 0.2023]).view(3, 1, 1)
        ) + torch.tensor([0.5071, 0.4865, 0.4409]).view(3, 1, 1)
        plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))

    plt.title("Input image")
    plt.show()


def plot_fgsm(
    model,
    model_name,
    device,
    test_loader,
    dataset,
    epsilons=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
):
    """
    Test the model's accuracy under FGSM (Fast Gradient Sign Method) attacks with different epsilon values,
    and plot the results.

    Parameters
    ----------
    model : torch.nn.Module
        The trained PyTorch model.
    model_name: str
        The name of the model.
    device : str
        The device to run the model on ('cpu' or 'cuda').
    test_loader : torch.utils.data.DataLoader
        The DataLoader for the test dataset.
    dataset: str
        The dataset the model is trained on.
    epsilons : list, optional
        A list of epsilon values for the FGSM attacks. Epsilon determines the step size of the perturbations
        during the attack. Default is [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3].

    This function will test the model's robustness against FGSM attacks by calculating its accuracy on the test
    dataset after each attack. Then, it will plot a graph of the accuracy results as a function of the epsilon
    values. The xticks and yticks on the graph are dynamically determined based on the range of epsilon values
    and accuracies, respectively.
    """
    filepath = f"./attacked_{dataset}_models/{model_name}_fgsm_accuracies.pkl"

    if os.path.exists(filepath):
        # Load the results if they have been previously calculated and saved
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            epsilons = data["epsilons"]
            accuracies = data["accuracies"]
    else:
        # Define the epsilon values to use for the FGSM attacks
        accuracies = []

        # Test the model's accuracy under FGSM attacks with each epsilon value
        for eps in epsilons:
            acc = fgsm_attack_test(model, device, test_loader, eps, dataset)
            accuracies.append(acc)

    # Calculate suitable step sizes for xticks
    xstep = (max(epsilons) - min(epsilons)) / (len(epsilons) - 1)

    # Plot the accuracy results
    plt.figure(figsize=(8, 6))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(min(epsilons), max(epsilons) + xstep, step=xstep))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()


def plot_multiple_fgsm(
    models,
    model_names,
    device,
    test_loader,
    dataset,
    epsilons=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
):
    """
    Test the models' accuracy under FGSM attacks with different epsilon values and plot the results.

    Parameters:
    models (dict): The dictionary of models to attack.
    model_names (list): The names of the models.
    device (torch.device): The device to perform computations on.
    test_loader (torch.utils.data.DataLoader): The test data loader.
    dataset (str): The dataset the models were trained on.
    epsilons (list, optional): A list of epsilon values for the FGSM attacks.

    This function will test each model's robustness against FGSM attacks by calculating its accuracy on the test
    dataset after each attack. Then, it will plot a graph of the accuracy results as a function of the epsilon
    values. The xticks and yticks on the graph are dynamically determined based on the range of epsilon values
    and accuracies, respectively.
    """
    colors = [
        "blue",
        "orange",
        "green",
        "red",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
        "lime",
        "navy",
    ]  # Add more colors if needed

    plt.figure(figsize=(10, 7))  # Increase the size of the plot for readability

    # Set the current color cycle
    plt.gca().set_prop_cycle("color", colors)

    for model_name in model_names:
        model = models[model_name].model
        filepath = f"./attacked_{dataset}_models/{model_name}_fgsm_accuracies.pkl"

        if os.path.exists(filepath):
            # Load the results if they have been previously calculated and saved
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                epsilons = data["epsilons"]
                accuracies = data["accuracies"]
        else:
            accuracies = []  # List to store results

            # Test the model's accuracy under PGD attacks with each number of iterations
            for eps in epsilons:
                acc = fgsm_attack_test(model, device, test_loader, eps, dataset)
                accuracies.append(acc)

            # Save the accuracies for all iterations for this model
            with open(filepath, "wb") as f:
                pickle.dump({"epsilons": epsilons, "accuracies": accuracies}, f)

        # Plot the results for each model
        plt.plot(epsilons, accuracies, "-*", label=model_name)

    # Calculate suitable step sizes for xticks
    xstep = (max(epsilons) - min(epsilons)) / (len(epsilons) - 1)

    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(min(epsilons), max(epsilons) + xstep, step=xstep))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.legend()  # Add a legend to distinguish lines for different models
    plt.grid(True)  # Add a grid for better visualization
    plt.show()


def plot_pgd(
    model,
    model_name,
    device,
    test_loader,
    dataset,
    eps=0.2,
    alpha=0.1,
    iters_list=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
):
    """
    Test the model's accuracy under PGD attacks with different number of iterations and plot the results.

    Parameters:
    model (torch.nn.Module): The model to attack.
    device (torch.device): The device to perform computations on.
    test_loader (torch.utils.data.DataLoader): The test data loader.
    eps (float): The maximum perturbation for each pixel.
    alpha (float): The step size for each iteration.
    iters_list (list, optional): A list of iteration counts for the PGD attacks.

    This function will test the model's robustness against PGD attacks by calculating its accuracy on the test
    dataset after each attack. Then, it will plot a graph of the accuracy results as a function of the iteration
    counts. The xticks and yticks on the graph are dynamically determined based on the range of iteration counts
    and accuracies, respectively.
    """
    filepath = f"./attacked_{dataset}_models/{model_name}_pgd_accuracies.pkl"

    if os.path.exists(filepath):
        # Load the results if they have been previously calculated and saved
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            iters_list = data["iters_list"]
            accuracies = data["accuracies"]
    else:
        accuracies = []  # List to store results

        # Test the model's accuracy under PGD attacks with each number of iterations
        for iters in iters_list:
            acc = pgd_attack_test(
                model, device, test_loader, eps, alpha, iters, dataset
            )
            accuracies.append(acc)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(iters_list, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(min(iters_list), max(iters_list) + 1, step=5))
    plt.title("Accuracy vs PGD Iterations")
    plt.xlabel("PGD Iterations")
    plt.ylabel("Accuracy")
    plt.show()


def plot_multiple_pgd(
    models,
    model_names,
    device,
    test_loader,
    dataset,
    eps=0.2,
    alpha=0.1,
    iters_list=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
):
    """
    Test the models' accuracy under PGD attacks with different number of iterations and plot the results.

    Parameters:
    models (dict): The models to attack.
    model_names (list): The names of the models.
    device (torch.device): The device to perform computations on.
    test_loader (torch.utils.data.DataLoader): The test data loader.
    dataset (str): The name of the dataset.
    eps (float): The maximum perturbation for each pixel.
    alpha (float): The step size for each iteration.
    iters_list (list, optional): A list of iteration counts for the PGD attacks.

    This function will test each models' robustness against PGD attacks by calculating its accuracy on the test
    dataset after each attack. Then, it will plot a graph of the accuracy results as a function of the iteration
    counts. The xticks and yticks on the graph are dynamically determined based on the range of iteration counts
    and accuracies, respectively.
    """
    colors = [
        "blue",
        "orange",
        "green",
        "red",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
        "lime",
        "navy",
    ]  # Add more colors if needed

    plt.figure(figsize=(10, 7))  # Increase the size of the plot for readability
    # Set the current color cycle
    plt.gca().set_prop_cycle("color", colors)

    for model_name in model_names:
        model = models[model_name].model
        filepath = f"./attacked_{dataset}_models/{model_name}_pgd_accuracies.pkl"

        if os.path.exists(filepath):
            # Load the results if they have been previously calculated and saved
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                iters_list = data["iters_list"]
                accuracies = data["accuracies"]
        else:
            accuracies = []  # List to store results

            # Test the model's accuracy under PGD attacks with each number of iterations
            for iters in iters_list:
                acc = pgd_attack_test(
                    model, device, test_loader, eps, alpha, iters, dataset
                )
                accuracies.append(acc)

            # Save the accuracies for all iterations for this model
            with open(filepath, "wb") as f:
                pickle.dump({"iters_list": iters_list, "accuracies": accuracies}, f)

        # Plot the results for each model
        plt.plot(iters_list, accuracies, "-*", label=model_name)

    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(min(iters_list), max(iters_list) + 1, step=5))
    plt.title("Accuracy vs PGD Iterations")
    plt.xlabel("PGD Iterations")
    plt.ylabel("Accuracy")
    plt.legend()  # Add a legend to distinguish lines for different models
    plt.grid(True)  # Add a grid for better visualization
    plt.show()


def plot_decision_boundaries_for_multiple_models(
    models_with_names,
    dataset,
    img,
    v1,
    v2,
    device,
    resolution=250,
    zoom=0.025,
    title=None,
    saveas=None,
):
    """
    Plots the decision boundaries of multiple models for a given image.

    Parameters:
    - models_with_names (list): List of tuples containing trained model and its name.
    - dataset (str): The dataset name the models were trained on.
    - img (torch.Tensor): The input image.
    - v1, v2 (torch.Tensor): Vectors to construct the 2D plane.
    - device (str): The device for computation ('cuda' or 'cpu').
    - resolution (int, optional): The resolution for visualization. Defaults to 250.
    - zoom (float, optional): The zoom level. Defaults to 0.025.
    - title (str, optional): Optional title prefix. Defaults to None.
    """

    if title is None:
        title = f"Decision Boundaries for Models Trained on {dataset}"

    num_models = len(models_with_names)
    assert 4 <= num_models <= 6, "Function supports only 4, 5, or 6 models."

    if num_models == 4:
        nrows, ncols = 2, 2
    elif num_models == 5:
        nrows, ncols = 2, 3  # we'll leave one spot empty
    else:  # 6 models
        nrows, ncols = 2, 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 8 * nrows))
    if num_models == 5:
        fig.delaxes(axes[1, 2])  # Delete the last axis for 5 models scenario

    # Flatten the image and move to device
    img = img.view(-1).to(device).float()

    scale = 1 / zoom
    x = torch.linspace(-scale, scale, resolution)
    y = torch.linspace(-scale, scale, resolution)
    xv, yv = torch.meshgrid(x, y)

    plane = (
        img[None, None, :]
        + xv[..., None] * v1[None, None, :]
        + yv[..., None] * v2[None, None, :]
    ).to(device)

    for idx, (model, model_name) in tqdm(enumerate(models_with_names)):
        row = idx // ncols
        col = idx % ncols
        model.to(device)
        model.eval()

        with torch.no_grad():
            if dataset == "mnist":
                output = model(plane.view(-1, 1, 28, 28)).view(
                    resolution, resolution, -1
                )
            elif dataset == "cifar10" or dataset == "cifar100":
                output = model(plane.view(-1, 3, 32, 32)).view(
                    resolution, resolution, -1
                )

        probs = F.softmax(output, dim=-1)
        _, predictions = torch.max(probs, dim=-1)
        decision_boundary = ndi.morphology.distance_transform_edt(
            predictions.cpu().numpy()
            == predictions.cpu().numpy()[resolution // 2, resolution // 2]
        )
        distance_to_boundary = (
            decision_boundary[resolution // 2, resolution // 2] / resolution * 2 * scale
        )

        colors = plt.get_cmap("tab10").colors
        cmap = ListedColormap([colors[i] for i in range(10)])

        ax = axes[row, col]
        img_plot = ax.imshow(
            predictions.cpu(),
            origin="lower",
            extent=(-scale, scale, -scale, scale),
            cmap=cmap,
        )
        ax.plot(0, 0, "ro")
        circle = plt.Circle((0, 0), distance_to_boundary, color="black", fill=False)
        ax.add_patch(circle)

        # Assuming you have a function called `total_variation_isotropic`
        tv = total_variation_isotropic(predictions.cpu().numpy())
        ax.set_title(f"Model: {model_name}\nTotal Variation (isotropic): {tv}")

    plt.tight_layout()
    plt.suptitle(title, fontsize=20, y=1.05)  # Adjust title position slightly above
    # plt.show()
    plt.savefig(
        f"{saveas}.png", dpi=300
    )  # You can set the desired dpi for better resolution


def plot_multiple_pgd_with_labels(
    models,
    model_name_labels,  # Changed this from model_names
    device,
    test_loader,
    dataset,
    eps=0.2,
    alpha=0.1,
    iters_list=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    title=None,  # New parameter
):
    """
    Test the models' accuracy under PGD attacks with different number of iterations and plot the results.

    Parameters:
    - models (dict): The models to attack.
    - model_name_labels (list): A list of tuples where each tuple contains (model_name, model_label).
    - device (torch.device): The device to perform computations on.
    - test_loader (torch.utils.data.DataLoader): The test data loader.
    - dataset (str): The name of the dataset.
    - eps (float): The maximum perturbation for each pixel.
    - alpha (float): The step size for each iteration.
    - iters_list (list, optional): A list of iteration counts for the PGD attacks.
    - suptitle (str, optional): An optional title for the entire plot.

    This function will test each models' robustness against PGD attacks by calculating its accuracy on the test
    dataset after each attack. Then, it will plot a graph of the accuracy results as a function of the iteration
    counts. The xticks and yticks on the graph are dynamically determined based on the range of iteration counts
    and accuracies, respectively.
    """
    colors = [
        "blue",
        "orange",
        "green",
        "red",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
        "lime",
        "navy",
    ]  # Add more colors if needed

    plt.figure(figsize=(10, 7))  # Increase the size of the plot for readability
    # Set the current color cycle
    plt.gca().set_prop_cycle("color", colors)

    for model_name, model_label in model_name_labels:  # Updated loop
        model = models[model_name].model
        filepath = f"./attacked_{dataset}_models/{model_name}_pgd_accuracies.pkl"

        if os.path.exists(filepath):
            # Load the results if they have been previously calculated and saved
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                iters_list = data["iters_list"]
                accuracies = data["accuracies"]
        else:
            accuracies = []  # List to store results

            # Test the model's accuracy under PGD attacks with each number of iterations
            for iters in iters_list:
                acc = pgd_attack_test(
                    model, device, test_loader, eps, alpha, iters, dataset
                )
                accuracies.append(acc)

            # Save the accuracies for all iterations for this model
            with open(filepath, "wb") as f:
                pickle.dump({"iters_list": iters_list, "accuracies": accuracies}, f)

        # Plot the results for each model using model_label for the legend
        plt.plot(iters_list, accuracies, "-*", label=model_label)

    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(min(iters_list), max(iters_list) + 1, step=5))
    plt.title(f"{title}")
    plt.xlabel("PGD Iterations")
    plt.ylabel("Accuracy")
    plt.legend()  # Add a legend to distinguish lines for different models
    plt.grid(True)  # Add a grid for better visualization
    plt.tight_layout()  # Adjusts the plot so that everything fits without overlap
    plt.show()


def plot_multiple_fgsm_with_labels(
    models,
    model_name_labels,  # Changed this from model_names
    device,
    test_loader,
    dataset,
    epsilons=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
    title=None,  # New parameter
):
    """
    Test the models' accuracy under FGSM attacks with different epsilon values and plot the results.

    Parameters:
    - models (dict): The dictionary of models to attack.
    - model_name_labels (list): A list of tuples where each tuple contains (model_name, model_label).
    - device (torch.device): The device to perform computations on.
    - test_loader (torch.utils.data.DataLoader): The test data loader.
    - dataset (str): The dataset the models were trained on.
    - epsilons (list, optional): A list of epsilon values for the FGSM attacks.
    - suptitle (str, optional): An optional title for the entire plot.

    This function will test each model's robustness against FGSM attacks by calculating its accuracy on the test
    dataset after each attack. Then, it will plot a graph of the accuracy results as a function of the epsilon
    values. The xticks and yticks on the graph are dynamically determined based on the range of epsilon values
    and accuracies, respectively.
    """
    colors = [
        "blue",
        "orange",
        "green",
        "red",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
        "lime",
        "navy",
    ]  # Add more colors if needed

    plt.figure(figsize=(10, 7))  # Increase the size of the plot for readability

    # Set the current color cycle
    plt.gca().set_prop_cycle("color", colors)

    for model_name, model_label in model_name_labels:  # Updated loop
        model = models[model_name].model
        filepath = f"./attacked_{dataset}_models/{model_name}_fgsm_accuracies.pkl"

        if os.path.exists(filepath):
            # Load the results if they have been previously calculated and saved
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                epsilons = data["epsilons"]
                accuracies = data["accuracies"]
        else:
            accuracies = []  # List to store results

            # Test the model's accuracy under FGSM attacks with each epsilon value
            for eps in epsilons:
                acc = fgsm_attack_test(model, device, test_loader, eps, dataset)
                accuracies.append(acc)

            # Save the accuracies for all epsilon values for this model
            with open(filepath, "wb") as f:
                pickle.dump({"epsilons": epsilons, "accuracies": accuracies}, f)

        # Plot the results for each model using model_label for the legend
        plt.plot(epsilons, accuracies, "-*", label=model_label)

    # Calculate suitable step sizes for xticks
    xstep = (max(epsilons) - min(epsilons)) / (len(epsilons) - 1)

    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(min(epsilons), max(epsilons) + xstep, step=xstep))
    plt.title(f"{title}")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.legend()  # Add a legend to distinguish lines for different models
    plt.grid(True)  # Add a grid for better visualization
    plt.tight_layout()  # Adjusts the plot so that everything fits without overlap
    plt.show()
