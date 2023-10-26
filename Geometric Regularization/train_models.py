"""
This script focuses on training different configurations of models on specified datasets and then saving both the trained model weights and the training metadata. The models are initialized with various configurations to evaluate different regularization and training techniques.

Main Script Execution:
    1. Sets up the device for computations (GPU if available, otherwise CPU).
    2. Chooses the dataset for training.
    3. Loads the train and test data for the selected dataset.
    4. Defines hyperparameters, specifically for L2 regularization (lambda) and dropout rate.
    5. Initializes different model configurations, focusing on different regularization techniques and dropout variations.
    6. Specifies the number of training epochs.
    7. Iteratively trains each model, displaying the progress.
    8. Checks if there are multiple GPUs available, and if so, leverages all of them for training using DataParallel.
    9. Trains the model, and after training, saves the model weights to a specified directory.
    10. Serializes and saves training metadata such as losses, regularized losses, epoch numbers, train, and test accuracies to disk using pickle.

Usage:
    Run this script to train various model configurations on the specified dataset, then save the models and training metadata for further analysis or deployment.
"""

import torch
import pickle
from data_generators import data_loader_mnist, data_loader_cifar10, data_loader_cifar100
from model_classes import LeNet, ResNet18
from tools import train
from torch.nn import DataParallel

if __name__ == "__main__":
    # Device configuration, use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set dataset
    dataset = "mnist"
    # dataset = "cifar10"
    # dataset = "cifar100"

    if dataset == "mnist":
        train_loader, test_loader = data_loader_mnist()
    elif dataset == "cifar10":
        train_loader, test_loader = data_loader_cifar10()
    elif dataset == "cifar100":
        train_loader, test_loader = data_loader_cifar100()

    # Hyperparameters are set in class init, except for dropout_rate and l2_lmbd
    l2_lmbd = 0.0005
    dropout_rate = 0.5

    # Initialize all models and store them in a dictionary with their names
    # Examples
    # For MNIST on LeNet:
    # "model_no_reg": LeNet_MNIST()
    # "model_no_reg_no_dropout": LeNet_MNIST(dropout_rate = 0.0)
    # "model_l2": LeNet_MNIST(l2_lmbd = 0.0005
    # "model_jacobi/svb": LeNet_MNIST(svb = True)
    # For CIFAR10 on DDNet, same as LeNet_MNIST, but lr = 0.01:
    # "model_no_reg": DDNet(),
    # For CIFAR100 on DDNET, same as above, except add keyword first:
    # "model_no_reg": DDNET(dataset = "cifar100")
    # For CIFAR100 on ResNet18, same as above, but dropout automatically set to 0:
    # "model_no_reg_0": ResNet18()

    models = {
        "model_no_reg": LeNet(),
        "model_dropout": LeNet(dropout_rate=0.5),
        "model_l2": LeNet(l2_lmbd=0.0005),
        "model_jacobi": LeNet(jacobi=True),
        "model_svb": LeNet(svb=True),
    }

    # Number of epochs for training, 250/300 in Hoffman 2019 for MNIST/CIFAR10
    n_epochs = 25

    # Iterate through each model
    for model_name, model in models.items():
        print(f"Training model: {model_name} on dataset: {dataset}")

        # Check if there are multiple GPUs, and if so, use DataParallel
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = DataParallel(model)

        # Move the model to the device (CPU or GPU)
        model.to(device)

        # Train model
        losses, reg_losses, epochs, train_accuracies, test_accuracies = train(
            train_loader, test_loader, model, device, n_epochs
        )

        # Save trained model
        torch.save(model.state_dict(), f"./trained_{dataset}_models/{model_name}.pt")

        # Save losses, reg_losses, epochs, train_accuracies, test_accuracies using pickle
        data = {
            "losses": losses,
            "reg_losses": reg_losses,
            "epochs": epochs,
            "train_accuracies": train_accuracies,
            "test_accuracies": test_accuracies,
        }

        with open(f"./trained_{dataset}_models/{model_name}_data.pkl", "wb") as f:
            pickle.dump(data, f)
