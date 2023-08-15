import torch
import torch.nn.functional as F
from tqdm import tqdm


def fgsm_attack(image, device, epsilon, data_grad, dataset):
    """
    Implements the Fast Gradient Sign Method (FGSM) attack.

    This function perturbs an input image by adding a small amount of noise in the direction
    of the loss gradient with respect to the input image. This perturbation aims to fool the model
    into misclassifying the perturbed image. Before applying the perturbation, the image is inversely
    normalized to its original scale, and after applying the perturbation, the image is normalized back
    to the scale the model expects. The magnitude of the perturbation is controlled by the epsilon parameter.

    Parameters:
    - image (torch.Tensor): The original, unperturbed image of shape (C, H, W), where C is the number
                            of channels, H is the height, and W is the width.
    - device (torch.device): The device (CPU/GPU) where computations will be performed.
    - epsilon (float): The perturbation magnitude. A small value that determines how much
                       to perturb the image in the direction of the gradient.
    - data_grad (torch.Tensor): The gradient of the loss with respect to the input image. It should have
                                the same shape as the input image.
    - dataset (str): The name of the dataset ("mnist", "cifar10", or "cifar100"). This is used to determine
                     the normalization and denormalization values for the image.

    Returns:
    - torch.Tensor: The perturbed image, which has undergone the FGSM attack. It retains the same shape
                    as the input image.

    Note:
    The function assumes specific normalization values based on the dataset. If these normalization values
    change or if new datasets are introduced, this function would need to be updated accordingly.
    """
    image = image.to(device)
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Inverse the original normalization
    if dataset == "mnist":
        inversed_image = image * 0.3081 + 0.1307
    elif dataset == "cifar10":
        inversed_image = image * 0.5 + 0.5
    elif dataset == "cifar100":
        inversed_image = (
            image * torch.tensor([0.2009, 0.1984, 0.2023]).view(3, 1, 1).to(device)
        ) + torch.tensor([0.5071, 0.4865, 0.4409]).view(3, 1, 1).to(device)
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
            perturbed_image
            - torch.tensor([0.5071, 0.4865, 0.4409]).view(3, 1, 1).to(device)
        ) / torch.tensor([0.2009, 0.1984, 0.2023]).view(3, 1, 1).to(device)
    # Return the perturbed image
    return perturbed_image


def fgsm_attack_test(model, device, test_loader, epsilon, dataset):
    """
    Evaluates the performance of a model under the Fast Gradient Sign Method (FGSM) attack.

    This function perturbs each image in the test dataset using the FGSM attack and evaluates
    the model's performance (accuracy) on these perturbed images. The primary purpose is to assess
    the robustness of the model against adversarial examples created using FGSM.

    Parameters:
    - model (torch.nn.Module): The PyTorch model to evaluate. It should be pretrained and ready
                               for evaluation.
    - device (torch.device): The device (CPU/GPU) where computations will be performed.
    - test_loader (torch.utils.data.DataLoader): DataLoader providing batches of the test dataset.
    - epsilon (float): The perturbation magnitude for the FGSM attack. Determines the strength of
                       the adversarial attack.
    - dataset (str): The name of the dataset ("mnist", "cifar10", or "cifar100"). This is used to
                     determine the normalization and denormalization values for the images.

    Returns:
    - float: The accuracy of the model on the perturbed test data. A value between 0 and 1, where
             1 indicates that the model correctly classified all perturbed images and 0 indicates
             that it failed to correctly classify any.

    Note:
    - The function assumes that the model returns the raw scores (logits) for each class, which is
      why a log_softmax is applied before computing the negative log likelihood loss.
    - The FGSM attack requires the gradient of the loss with respect to the input, which is why
      `data.requires_grad` is set to True.
    """

    # Accuracy counter
    correct = 0
    model.eval()
    model = model.to(device)

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
        perturbed_data = fgsm_attack(data, device, epsilon, data_grad, dataset)
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
                images * torch.tensor([0.2009, 0.1984, 0.2023]).view(3, 1, 1).to(device)
            ) + torch.tensor([0.5071, 0.4865, 0.4409]).view(3, 1, 1).to(device)

        # Clip values to [0,1]
        images = torch.clamp(images, min=0, max=1)

        # Reapply normalization after clipping
        if dataset == "mnist":
            images = (images - 0.1307) / 0.3081
        elif dataset == "cifar10":
            images = (images - 0.5) / 0.5
        elif dataset == "cifar100":
            images = (
                images - torch.tensor([0.5071, 0.4865, 0.4409]).view(3, 1, 1).to(device)
            ) / torch.tensor([0.2009, 0.1984, 0.2023]).view(3, 1, 1).to(device)

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
    dataset (str): The dataset the images belong to.

    Returns:
    float: The accuracy of the model on perturbed test data.
    """
    # Counter for correct predictions
    correct = 0
    model.eval()
    model = model.to(device)

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
