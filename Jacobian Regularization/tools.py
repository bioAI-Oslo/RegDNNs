import torch
import torch.nn as nn
from tqdm import tqdm


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
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
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
                for g in model.opt.param_groups:
                    g["lr"] = g["lr"] / 10
                    print(f"Decayed lr from {g['lr'] * 10} to {g['lr']}")

            iterations += 1

        train_accuracies.append(accuracy(model, test_loader, device))
        test_accuracies.append(accuracy(model, train_loader, device))
        model.counter = 0
        print(f"Epoch: {epoch}")
        print(
            "Accuracy of the network on the test images: %.2f %%"
            % (100 * accuracy(model, test_loader, device))
        )
    return losses, reg_losses, epochs, weights, train_accuracies, test_accuracies


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
