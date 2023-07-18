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

        if model.hard_svb:
            svb(model, eps=model.hard_svb_lmbd)

        train_accuracies.append(accuracy(model, test_loader, device))
        test_accuracies.append(accuracy(model, train_loader, device))
        model.counter = 0
        print(f"Epoch: {epoch}")
        print(
            "Accuracy of the network on the test images: %d %%"
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
