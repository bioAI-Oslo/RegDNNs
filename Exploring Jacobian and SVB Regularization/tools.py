import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import io
from tqdm import tqdm
from collections import OrderedDict

from model_classes import LeNet_MNIST


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

            if device == "cuda":
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
                for g in model.opt.param_groups:
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
    return losses, reg_losses, epochs, weights, train_accuracies, test_accuracies


def accuracy(model, loader, device):
    """Calculate the accuracy of a model. Uses a data loader."""
    correct = 0
    total = 0
    model.eval()  # switch to evaluation mode
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()  # switch back to training mode
    return correct / total


def train_remote(
    train_loader,
    test_loader,
    model,
    device,
    n_epochs=2,
):
    losses = []
    epochs = []
    train_accuracies = []
    test_accuracies = []
    reg_losses = []
    iterations = 0

    for epoch in tqdm(range(n_epochs)):
        N = len(train_loader)
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

        train_accuracies.append(accuracy(model, test_loader, device))
        test_accuracies.append(accuracy(model, train_loader, device))
        print(f"Epoch: {epoch+1}")
        print(
            "Accuracy of the network on the test images: %.2f %%"
            % (100 * accuracy(model, test_loader, device))
        )
    return losses, reg_losses, epochs, train_accuracies, test_accuracies


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


def load_model(model_name):
    device = torch.device("cpu")

    if model_name == "model_no_reg":
        model = (
            LeNet_MNIST()
        )  # Initialize your model here. Make sure it matches the architecture of the saved model.
    elif model_name == "model_l2":
        model = LeNet_MNIST(l2_lmbd=0.0005)
    elif model_name == "model_jacobi":
        model = LeNet_MNIST(jacobi_reg=True, jacobi_reg_lmbd=0.01)
    elif model_name == "model_svb":
        model = LeNet_MNIST(svb_reg=True, svb_freq=100, svb_eps=0.01)

    # Load state dict
    state_dict = torch.load(f"./trained_models/{model_name}.pt", map_location=device)

    # Create new OrderedDict that does not contain `module.` prefix
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    # Load parameters
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    with open(f"./trained_models/{model_name}_data.pkl", "rb") as f:
        data = pickle.load(f)

    losses = data["losses"]
    reg_losses = data["reg_losses"]
    epochs = data["epochs"]
    train_accuracies = data["train_accuracies"]
    test_accuracies = data["test_accuracies"]

    return model, losses, reg_losses, epochs, train_accuracies, test_accuracies


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def fgsm_attack_test(model, device, test_loader, epsilon):
    # Accuracy counter
    correct = 0
    adv_examples = []
    # Loop over all examples in test set
    for data, target in test_loader:
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True
        # Forward pass the data through the model
        output = model(data)
        # Calculate the loss
        loss = F.nll_loss(F.log_softmax(output, dim=1), target)
        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        loss.backward()
        # Collect datagrad
        data_grad = data.grad.data
        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        # Re-classify the perturbed image
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[
            1
        ]  # get the index of the max log-probability
        correct += (
            (final_pred.view(target.shape) == target).sum().item()
        )  # update the correct predictions count

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(test_loader.dataset))
    return final_acc