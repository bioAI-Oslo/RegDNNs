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
    l2_lmbd = 0.0005
    jacobi_reg_lmbd = 0.01

    # Initialize model
    model = LeNet_MNIST(jacobi_reg=True, jacobi_reg_lmbd=jacobi_reg_lmbd)

    # Check if there are multiple GPUs, and if so, use DataParallel
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = DataParallel(model)

    # Move the model to the device (CPU or GPU)
    model.to(device)

    n_epochs = 75
    (
        losses,
        reg_losses,
        epochs,
        weights,
        train_accuracies,
        test_accuracies,
    ) = train_remote(train_loader, test_loader, model, device, n_epochs)

    if not os.path.exists("./trained_models"):
        os.makedirs("./trained_models")

    torch.save(
        model.state_dict(),
        "./trained_models/model_jacobi.pt",
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

    with open("./trained_models/model_jacobi_data.pkl", "wb") as f:
        pickle.dump(data, f)
