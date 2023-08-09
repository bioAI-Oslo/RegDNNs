import numpy as np
import pickle
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import OrderedDict

from model_classes import LeNet_MNIST, DDNet, ResNet18


def train(
    train_loader,
    test_loader,
    model,
    device,
    n_epochs,
):
    """
    Function to train a PyTorch model.

    Parameters:
    train_loader (DataLoader): DataLoader for the training data.
    test_loader (DataLoader): DataLoader for the test data.
    model (nn.Module): The PyTorch model to train.
    device (string): The device to run training on. Usually "cuda" or "cpu".
    n_epochs (int): The number of epochs to train the model for.

    Returns:
    Tuple of lists: Tracking of losses, regularization losses, epochs,
    training accuracies, and testing accuracies over training process.
    """
    losses = []
    epochs = []
    train_accuracies = []
    test_accuracies = []
    reg_losses = []
    iterations = 0

    # Train on dataset epoch times
    for epoch in tqdm(range(n_epochs)):
        N = len(train_loader)

        # Loop over data in train_loader
        for i, (data, labels) in enumerate(train_loader):
            epochs.append(epoch + i / N)
            data = data.to(device)
            labels = labels.to(device)

            # Compute loss based on # available GPUs
            if torch.cuda.device_count() > 1:
                loss_data, reg_loss_data = model.module.train_step(
                    data,
                    labels,
                )
            else:
                loss_data, reg_loss_data = model.train_step(
                    data,
                    labels,
                )
            losses.append(loss_data)
            reg_losses.append(reg_loss_data)

            # Learning rate decay
            if iterations % (n_epochs * 200) == 0 and iterations > 0:
                if torch.cuda.device_count() > 1:
                    opt = model.module.opt
                else:
                    opt = model.opt

                # Update lr for each parameter group
                for g in opt.param_groups:
                    g["lr"] = g["lr"] / 10
                    print(f"Decayed lr from {g['lr'] * 10} to {g['lr']}")

            iterations += 1

        train_accuracies.append(accuracy(model, train_loader, device))
        test_accuracies.append(accuracy(model, test_loader, device))
        print(f"Epoch: {epoch+1}")
        print(
            "Accuracy of the network on the test images: %.2f %%"
            % (100 * accuracy(model, test_loader, device))
        )
    return losses, reg_losses, epochs, train_accuracies, test_accuracies


def accuracy(model, loader, device):
    """Calculate the accuracy of a model.

    Parameters:
    model (nn.Module): The PyTorch model to evaluate.
    loader (DataLoader): DataLoader for the dataset to evaluate on.
    device (str): The device to run evaluation on. Usually "cuda" or "cpu".

    Returns:
    float: The accuracy of the model on the given dataset.
    """
    correct = 0
    total = 0

    # Switch to evaluation mode
    model.eval()

    # Do not track gradients
    with torch.no_grad():
        # Calculate accuracy
        for data in loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # Switch back to training mode
    model.train()
    return correct / total


class SaveOutput:
    """
    Class to save outputs from specified layers in a model.
    """

    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        """
        Saves the output from a forward pass through a module.

        Parameters:
        module (nn.Module): The module that just performed a forward pass.
        module_in (Tensor): The input to the module.
        module_out (Tensor): The output from the module.
        """
        self.outputs.append(module_out)

    def clear(self):
        """
        Clear the saved outputs.
        """
        self.outputs = []


def register_hooks(model):
    """
    Registers forward hooks for Conv2d and Linear layers in a model.

    Parameters:
    model (nn.Module): The PyTorch model to register hooks on.

    Returns:
    tuple: A tuple containing an instance of SaveOutput (to access saved outputs),
    a list of handles to the hooks (for removing the hooks later), and a list of
    the names of the layers hooks were registered on.
    """
    save_output = SaveOutput()
    layer_names = []

    # Register hooks for conv and fc layers
    hook_handles = []
    for name, layer in model._modules.items():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            handle = layer.register_forward_hook(save_output)
            hook_handles.append(handle)
            layer_names.append(name)

    return save_output, hook_handles, layer_names


def load_trained_model(model_name, dataset):
    """
    Loads a pre-trained PyTorch model and associated training data.

    Parameters:
    model_name (str): The name of the model to load. This is used to locate the saved model
                      and training data files.

    Returns:
    tuple: A tuple containing the loaded model and lists of losses, regularization losses,
           epochs, and train/test accuracies from training.
    """
    # Set to cpu as we will be loading on a laptop
    device = torch.device("cpu")

    # Initialize model based on provided model_name to get a similar model to prevent errors
    # FIX THIS!
    if dataset == "mnist":
        if model_name.startswith("model_no_reg"):
            model = LeNet_MNIST()
        elif model_name.startswith("model_l2"):
            model = LeNet_MNIST(l2_lmbd=0.0005)
        elif model_name.startswith("model_svb"):
            model = LeNet_MNIST(svb=True)
        elif model_name.startswith("model_jacobi"):
            model = LeNet_MNIST(jacobi=True)
        elif model_name.startswith("model_all"):
            model = LeNet_MNIST(l2_lmbd=0.0005, jacobi=True, svb=True)
    elif dataset == "cifar10":
        if model_name.startswith("model_no_reg"):
            model = DDNet()
        elif model_name.startswith("model_l2"):
            model = DDNet(l2_lmbd=0.0005)
        elif model_name.startswith("model_svb"):
            model = DDNet(svb=True)
        elif model_name.startswith("model_jacobi"):
            model = DDNet(jacobi=True)
        elif model_name.startswith("model_all"):
            model = DDNet(l2_lmbd=0.0005, jacobi=True, svb=True)
    elif dataset == "cifar100":
        if model_name.startswith("model_no_reg"):
            model = ResNet18()
        elif model_name.startswith("model_l2"):
            model = ResNet18(l2_lmbd=0.0005)
        elif model_name.startswith("model_svb"):
            model = ResNet18(svb=True)
        elif model_name.startswith("model_jacobi"):
            model = ResNet18(jacobi=True)
        elif model_name.startswith("model_all"):
            model = ResNet18(l2_lmbd=0.0005, jacobi=True, svb=True)
    else:
        print("Error: Dataset not implemented")

    # Load state dictionary
    state_dict = torch.load(
        f"./trained_{dataset}_models/{model_name}.pt", map_location=device
    )

    # Create new OrderedDict that does not contain `module.` prefix
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        # name = k[len("module.") :] if k.startswith("module.") else k
        new_state_dict[name] = v

    # Load parameters into model
    model.load_state_dict(new_state_dict)
    model.to(device)

    model.eval()

    # Load training data
    with open(f"./trained_{dataset}_models/{model_name}_data.pkl", "rb") as f:
        data = pickle.load(f)

    losses = data["losses"]
    reg_losses = data["reg_losses"]
    epochs = data["epochs"]
    train_accuracies = data["train_accuracies"]
    test_accuracies = data["test_accuracies"]

    return model, losses, reg_losses, epochs, train_accuracies, test_accuracies


class ModelInfo:
    """
    This class is used to store information about a trained model.

    Attributes:
    name (str): The name of the model.
    model (nn.Module): The PyTorch model.
    losses (list): The list of loss values during training.
    reg_losses (list): The list of regularization loss values during training.
    epochs (int): The number of training epochs.
    train_accuracies (list): The list of training accuracies.
    test_accuracies (list): The list of test accuracies.
    """

    def __init__(self, name, dataset):
        """
        The constructor for ModelInfo class.

        Parameters:
        name (str): The name of the model.
        """

        self.name = name
        self.dataset = dataset
        (
            self.model,
            self.losses,
            self.reg_losses,
            self.epochs,
            self.train_accuracies,
            self.test_accuracies,
        ) = self.load_model(name, dataset)

    @staticmethod
    def load_model(name, dataset):
        """
        This static method is used to load the model and related information from a file using load_trained_model.

        Parameters:
        name (str): The name of the model.

        Returns:
        model (nn.Module): The loaded PyTorch model.
        losses (list): The loaded list of loss values during training.
        reg_losses (list): The loaded list of regularization loss values during training.
        epochs (int): The loaded number of training epochs.
        train_accuracies (list): The loaded list of training accuracies.
        test_accuracies (list): The loaded list of test accuracies.
        """
        (
            model,
            losses,
            reg_losses,
            epochs,
            train_accuracies,
            test_accuracies,
        ) = load_trained_model(name, dataset)
        return model, losses, reg_losses, epochs, train_accuracies, test_accuracies


def total_variation_isotropic(image):
    """
    Function to compute the isotropic total variation of an image.

    Parameters
    ----------
    image : numpy.ndarray
        2D numpy array representing the grayscale image.

    Returns
    -------
    total_variation : float
        Isotropic total variation of the image.
    """
    diffs_in_x = np.diff(image, axis=1)[:-1, :]  # Slice to match y dimension after diff
    diffs_in_y = np.diff(image, axis=0)[:, :-1]  # Slice to match x dimension after diff

    magnitude_of_diffs = np.sqrt(diffs_in_x**2 + diffs_in_y**2)
    total_variation = np.sum(magnitude_of_diffs)
    return total_variation


def total_variation_anisotropic(image):
    """
    Function to compute the anisotropic total variation of an image.

    Parameters
    ----------
    image : numpy.ndarray
        2D numpy array representing the grayscale image.

    Returns
    -------
    total_variation : float
        Anisotropic total variation of the image.
    """
    diffs_in_x = np.abs(np.diff(image, axis=1))
    diffs_in_y = np.abs(np.diff(image, axis=0))

    total_variation = np.sum(diffs_in_x) + np.sum(diffs_in_y[:-1])
    return total_variation


def compute_total_variation(
    model,
    img,
    v1,
    v2,
    device,
    resolution=300,
    zoom=[0.025, 0.01, 0.001],
    mode="isotropic",
    dataset="mnist",
):
    """
    Function to calculate the total variation of decision boundaries of a model in a 2D plane for different zoom levels.

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
        The zoom levels to calculate total variation of, by default [0.025, 0.01, 0.001].
    mode : str, optional
        The mode of total variation to be used, can be either 'isotropic' or 'anisotropic', by default 'isotropic'.
    """

    # Check if the mode is valid
    if mode not in ["isotropic", "anisotropic"]:
        raise ValueError("Invalid mode. Expected 'isotropic' or 'anisotropic'")

    # Enter evaluation mode
    model.eval()

    # Flatten the image if necessary and convert to a single precision float
    img = img.view(-1).to(device).float()

    tv_list = []
    for zoom_level in zoom:
        # Generate grid of points in plane defined by img, v1 and v2
        scale = 1 / zoom_level  # Scale decided by zoom_level
        x = torch.linspace(-scale, scale, resolution).to(device)
        y = torch.linspace(-scale, scale, resolution).to(device)
        xv, yv = torch.meshgrid(x, y)

        v1 = v1.to(device)  # Move v1 to the same device as the model
        v2 = v2.to(device)  # Move v2 to the same device as the model

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

        # Compute the total variation according to the specified mode
        if mode == "isotropic":
            tv = total_variation_isotropic(predictions.cpu().numpy())
        else:
            tv = total_variation_anisotropic(predictions.cpu().numpy())
        tv_list.append(tv)
    return tv_list
