import torch
import os
import pickle
from data_generators import data_loader_MNIST
from model_classes import LeNet_MNIST
from tools import train_remote
from torch.nn import DataParallel

if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading MNIST dataset
    train_loader, test_loader = data_loader_MNIST()

    # Hyperparameters
    lr = 0.1
    momentum = 0.9

    # Initialize model
    model_no_reg = LeNet_MNIST()

    # Check if there are multiple GPUs, and if so, use DataParallel
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_no_reg = DataParallel(model_no_reg)

    # Move the model to the device (CPU or GPU)
    model_no_reg.to(device)

    n_epochs = 25
    (
        losses,
        reg_losses,
        epochs,
        weights,
        train_accuracies,
        test_accuracies,
    ) = train_remote(train_loader, test_loader, model_no_reg, device, n_epochs)

    if not os.path.exists("./Jacobian Regularization/trained_models"):
        os.makedirs("./Jacobian Regularization/trained_models")

    torch.save(
        model_no_reg.state_dict(),
        "./Jacobian Regularization/trained_models/model_no_reg.pt",
    )

    # Save losses, reg_losses, epochs, weights, train_accuracies, test_accuracies using pickle
    data = {
        "losses": losses,
        "reg_losses": reg_losses,
        "epochs": epochs,
        "weights": weights,
        "train_accuracies": train_accuracies,
        "test_accuracies": test_accuracies,
    }

    with open(
        "./Jacobian Regularization/trained_models/model_no_reg_data.pkl", "wb"
    ) as f:
        pickle.dump(data, f)
