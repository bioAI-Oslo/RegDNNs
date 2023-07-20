import torch
import pickle
import torch.nn as nn
from tqdm import tqdm
from collections import OrderedDict

from model_classes import LeNet


def train(
    train_loader,
    test_loader,
    model,
    device,
    n_epochs=2,
):
    losses = []
    epochs = []
    weights = []
    train_accuracies = []
    test_accuracies = []
    reg_losses = []
    iterations = 0

    for epoch in tqdm(range(n_epochs)):
        N = len(train_loader)
        for param in model.parameters():
            weights.append(param.detach().cpu().numpy().copy())
        for i, (data, labels) in enumerate(train_loader):
            epochs.append(epoch + i / N)
            data = data.to(device)
            labels = labels.to(device)

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
                for g in opt.param_groups:
                    g["lr"] = g["lr"] / 10
                    print(f"Decayed lr from {g['lr'] * 10} to {g['lr']}")

            iterations += 1

        train_accuracies.append(accuracy(model, train_loader, device))
        test_accuracies.append(accuracy(model, test_loader, device))
        model.counter = 0
        print(f"Epoch: {epoch+1}")
        print(
            "Accuracy of the network on the test images: %.2f %%"
            % (100 * accuracy(model, test_loader, device))
        )
    return losses, reg_losses, epochs, weights, train_accuracies, test_accuracies


def accuracy(model, loader, device):
    """Calculate the accuracy of a model. Uses a data loader."""
    correct = 0
    total = 0
    model.eval()  # Switch to evaluation mode
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()  # Switch back to training mode
    return correct / total


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


def register_hooks(model):
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


def load_trained_model(model_name):
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
    if model_name == "model_no_reg":
        model = LeNet()
    elif model_name == "model_l1":
        model = LeNet(l1=True)
    elif model_name == "model_l2":
        model = LeNet(l2=True)
    elif model_name == "model_l1_l2":
        model = LeNet(l1_l2=True)
    elif model_name == "model_svb":
        model = LeNet(svb=True)
    elif model_name == "model_soft_svb":
        model = LeNet(soft_svb=True)
    elif model_name == "model_jacobi_reg":
        model = LeNet(jacobi_reg=True)
    elif model_name == "model_jacobi_det_reg":
        model = LeNet(jacobi_det_reg=True)
    elif model_name == "model_dropout":
        model = LeNet(dropout_rate=0.5)
    elif model_name == "model_conf_penalty":
        model = LeNet(conf_penalty=True)
    elif model_name == "model_label_smoothing":
        model = LeNet(label_smoothing=True)
    elif model_name == "model_noise_inj_inputs":
        model = LeNet(noise_inject_inputs=True)
    elif model_name == "model_noise_inj_weights":
        model = LeNet(noise_inject_weights=True)

    # Load state dictionary
    state_dict = torch.load(
        f"./trained_mnist_models/{model_name}.pt", map_location=device
    )

    # Create new OrderedDict that does not contain `module.` prefix
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    # Load parameters into model
    model.load_state_dict(new_state_dict)
    model.to(device)

    model.eval()

    # Load training data
    with open(f"./trained_mnist_models/{model_name}_data.pkl", "rb") as f:
        data = pickle.load(f)

    losses = data["losses"]
    reg_losses = data["reg_losses"]
    epochs = data["epochs"]
    weights = data["weights"]
    train_accuracies = data["train_accuracies"]
    test_accuracies = data["test_accuracies"]

    return model, losses, reg_losses, epochs, weights, train_accuracies, test_accuracies


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

    def __init__(self, name):
        """
        The constructor for ModelInfo class.

        Parameters:
        name (str): The name of the model.
        """

        self.name = name
        (
            self.model,
            self.losses,
            self.reg_losses,
            self.epochs,
            self.weights,
            self.train_accuracies,
            self.test_accuracies,
        ) = self.load_model(name)

    @staticmethod
    def load_model(name):
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
            weights,
            train_accuracies,
            test_accuracies,
        ) = load_trained_model(name)
        return (
            model,
            losses,
            reg_losses,
            epochs,
            weights,
            train_accuracies,
            test_accuracies,
        )
