import torch
import os
import pickle
from data_generators import data_loader_MNIST
from model_classes import LeNet_MNIST
from tools import train
from torch.nn import DataParallel

if __name__ == "__main__":
    # Device configuration, use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading MNIST dataset
    train_loader, test_loader = data_loader_MNIST()

    # Hyperparameters for the models
    lr = 0.1
    momentum = 0.9
    l2_lmbd = 0.0005
    jacobi_reg_lmbd = 0.01
    svb_freq = 600
    svb_eps = 0.05

    # Number of epochs for training, 250 in Hoffman 2019
    n_epochs = 250

    # Create a directory to save the trained models, if it does not exist
    if not os.path.exists("./trained_mnist_models"):
        os.makedirs("./trained_mnist_models")

    # Train n models
    n = 4
    for i in range(1, n + 1):
        # Initialize model with regularization of choice
        model = LeNet_MNIST(
            dropout_rate=0.0, jacobi_reg=True, jacobi_reg_lmbd=jacobi_reg_lmbd
        )

        # Check if there are multiple GPUs, and if so, use DataParallel
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = DataParallel(model)

        # Move the model to the device
        model = model.to(device)

        # Train model and collect data
        (
            losses,
            reg_losses,
            epochs,
            train_accuracies,
            test_accuracies,
        ) = train(train_loader, test_loader, model, device, n_epochs)

        # Switch to evaluation mode
        model.eval()

        # Save state of trained model based on training
        if isinstance(model, torch.nn.DataParallel):
            torch.save(
                model.module.state_dict(),
                f"./trained_mnist_models/model_jacobi_no_dropout_{i}.pt",
            )
        else:
            torch.save(
                model.state_dict(),
                f"./trained_mnist_models/model_jacobi_no_dropout_{i}.pt",
            )

        # Save losses, reg_losses, epochs, train_accuracies, test_accuracies using pickle
        data = {
            "losses": losses,
            "reg_losses": reg_losses,
            "epochs": epochs,
            "train_accuracies": train_accuracies,
            "test_accuracies": test_accuracies,
        }

        with open(
            f"./trained_mnist_models/model_jacobi_no_dropout_{i}_data.pkl", "wb"
        ) as f:
            pickle.dump(data, f)