import torch
import pickle
from data_generators import data_loader_MNIST, data_loader_CIFAR10, data_loader_CIFAR100
from model_classes import LeNet_MNIST, DDNet
from tools import train
from torch.nn import DataParallel

if __name__ == "__main__":
    # Device configuration, use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set dataset
    # dataset = "mnist"
    dataset = "cifar10"
    # dataset = "cifar100"

    if dataset == "mnist":
        train_loader, test_loader = data_loader_MNIST()
    elif dataset == "cifar10":
        train_loader, test_loader = data_loader_CIFAR10()
    elif dataset == "cifar100":
        train_loader, test_loader = data_loader_CIFAR100()

    # Hyperparameters are set in class init, except for dropout_rate and l2_lmbd
    l2_lmbd = 0.0005
    dropout_rate = 0.5
    # For CIFAR10, based on my own testing
    lr = 0.05

    # Initialize all models and store them in a dictionary with their names
    models = {
        "model_no_reg_0": DDNet(lr=lr),
        "model_l2_0": DDNet(lr=lr, l2_lmbd=l2_lmbd),
        "model_svb_0": DDNet(lr=lr, svb=True),
        "model_jacobi_0": DDNet(lr=lr, jacobi=True),
        "model_jacobi_no_dropout_0": DDNet(lr=lr, dropout_rate=0.0, jacobi=True),
        # "model_no_reg_0": DDNet(dataset = "cifar100"),
        # "model_l2_0": DDNet(dataset = "cifar100", l2_lmbd=l2_lmbd),
        # "model_svb_0": DDNet(dataset = "cifar100", svb=True),
        # "model_jacobi_0": DDNet(dataset = "cifar100", jacobi=True),
        # "model_jacobi_no_dropout_0": DDNet(dataset = "cifar100", dropout_rate=0.0, jacobi=True),
    }

    # Number of epochs for training, 250 in Hoffman 2019
    n_epochs = 50

    # Iterate through each model
    for model_name, model in models.items():
        print(f"Training model: {model_name} on dataset: {dataset}")

        # Check if there are multiple GPUs, and if so, use DataParallel
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = DataParallel(model)

        # Move the model to the device (CPU or GPU)
        model.to(device)

        losses, reg_losses, epochs, train_accuracies, test_accuracies = train(
            train_loader, test_loader, model, device, n_epochs
        )

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
