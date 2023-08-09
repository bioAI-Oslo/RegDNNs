import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms


def fgsm_attack(image, epsilon, data_grad, dataset):
    """
    Implements the Fast Gradient Sign Method (FGSM) attack.

    Parameters:
    image (torch.Tensor): The original, unperturbed image.
    epsilon (float): The perturbation magnitude.
    data_grad (torch.Tensor): The gradient of the loss with respect to the input image.

    Returns:
    torch.Tensor: The perturbed image.
    """
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Inverse the original normalization
    if dataset == "mnist":
        inversed_image = image * 0.3081 + 0.1307
    elif dataset == "cifar10":
        inversed_image = image * 0.5 + 0.5
    elif dataset == "cifar100":
        inversed_image = (
            image * torch.tensor([0.2009, 0.1984, 0.2023]).view(3, 1, 1)
        ) + torch.tensor([0.5071, 0.4865, 0.4409]).view(3, 1, 1)
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = inversed_image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    # Apply the original normalization
    if dataset == "mnist":
        perturbed_image = (perturbed_image - 0.1307) / 0.3081
    elif dataset == "cifar10":
        perturbed_image = (perturbed_image - 0.5) / 0.5
    elif dataset == "cifar100":
        perturbed_image = (
            perturbed_image - torch.tensor([0.5071, 0.4865, 0.4409]).view(3, 1, 1)
        ) / torch.tensor([0.2009, 0.1984, 0.2023]).view(3, 1, 1)
    # Return the perturbed image
    return perturbed_image


def fgsm_attack_test(model, device, test_loader, epsilon, dataset):
    """
    Test a model under FGSM attack.

    Parameters:
    model (torch.nn.Module): The PyTorch model to evaluate.
    device (torch.device): The device to perform computations on.
    test_loader (torch.utils.data.DataLoader): The test data loader.
    epsilon (float): The perturbation magnitude for FGSM attack.

    Returns:
    float: The accuracy of the model on perturbed test data.
    """

    # Accuracy counter
    correct = 0
    model.eval()

    # Loop over all examples in test set
    for data, target in tqdm(test_loader):
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
        perturbed_data = fgsm_attack(data, epsilon, data_grad, dataset)
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


def pgd_attack(model, images, labels, device, eps, alpha, iters, dataset):
    """
    Perform the Projected Gradient Descent (PGD) attack.

    Parameters:
    model (torch.nn.Module): The model to attack.
    images (torch.Tensor): Original images.
    labels (torch.Tensor): True labels for `images`.
    device (torch.device): The device to perform computations on.
    eps (float): The maximum perturbation for each pixel.
    alpha (float): The step size for each iteration.
    iters (int): The number of iterations.
    dataset (str): The dataset the images belong to.

    Returns:
    torch.Tensor: The perturbed images.
    """
    # Move data to target device
    images = images.to(device)
    labels = labels.to(device)

    # Define the loss function
    loss = torch.nn.CrossEntropyLoss()

    # Keep a copy of the original images
    ori_images = images.data

    # Iteratively apply the PGD attack
    for j in tqdm(range(iters)):
        images.requires_grad = True  # Enable gradient calculation on images
        outputs = model(images)  # Forward pass

        model.zero_grad()
        cost = loss(outputs, labels).to(device)  # Compute loss
        cost.backward()  # Backward pass

        # Update images by adding a small step in the direction of the gradient
        adv_images = images + alpha * images.grad.sign()
        # Clip perturbations of original image to be within epsilon
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)

        # Add clipped perturbations to the original images to get the adverserial images
        images = (ori_images + eta).detach_()

        # Reverse normalization before clipping to ensure the perturbed images are in the correct range
        if dataset == "mnist":
            images = images * 0.3081 + 0.1307
        elif dataset == "cifar10":
            images = images * 0.5 + 0.5
        elif dataset == "cifar100":
            images = (
                images * torch.tensor([0.2009, 0.1984, 0.2023]).view(3, 1, 1)
            ) + torch.tensor([0.5071, 0.4865, 0.4409]).view(3, 1, 1)

        # Clip values to [0,1]
        images = torch.clamp(images, min=0, max=1)

        # Reapply normalization after clipping
        if dataset == "mnist":
            images = (images - 0.1307) / 0.3081
        elif dataset == "cifar10":
            images = (images - 0.5) / 0.5
        elif dataset == "cifar100":
            images = (
                images - torch.tensor([0.5071, 0.4865, 0.4409]).view(3, 1, 1)
            ) / torch.tensor([0.2009, 0.1984, 0.2023]).view(3, 1, 1)

    return images


def pgd_attack_test(model, device, test_loader, eps, alpha, iters, dataset):
    """
    Test the model under PGD attack and return the accuracy on perturbed images.

    Parameters:
    model (torch.nn.Module): The model to evaluate.
    device (torch.device): The device to perform computations on.
    test_loader (torch.utils.data.DataLoader): The test data loader.
    eps (float): The maximum perturbation for each pixel.
    alpha (float): The step size for each iteration.
    iters (int): The number of iterations.

    Returns:
    float: The accuracy of the model on perturbed test data.
    """
    # Counter for correct predictions
    correct = 0
    model.eval()

    # Iterate over the test dataset
    for images, labels in test_loader:
        # Move images and labels to target device
        images = images.to(device)
        labels = labels.to(device)

        # Generate adverserial examples using PGD
        perturbed_images = pgd_attack(
            model, images, labels, device, eps, alpha, iters, dataset
        )
        # Get predictions for the adverserial examples
        outputs = model(perturbed_images)

        # Get the index of the max log-probability as the predicted label
        _, predicted = torch.max(outputs.data, 1)
        # Update correct predictions counter
        correct += (predicted == labels).sum().item()

    # Compute the final accuracy of the model on the adversarial examples
    final_acc = correct / float(len(test_loader.dataset))
    return final_acc
