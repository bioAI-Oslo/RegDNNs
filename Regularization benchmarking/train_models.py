import torch
import pickle
from data_generators import data_loader_MNIST, data_loader_CIFAR10
from model_classes import LeNet, DDNet
from tools import train
from torch.nn import DataParallel

if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set dataset
    # dataset = "mnist"
    dataset = "mnist"

    if dataset == "mnist":
        train_loader, test_loader = data_loader_MNIST()
    elif dataset == "cifar10":
        train_loader, test_loader = data_loader_CIFAR10()

    # Hyperparameters are set in class init, except for dropout_rate
    dropout_rate = 0.5

    # Initialize all models and store them in a dictionary with their names
    models = {
        "model_no_reg": LeNet(),
        "model_l1": LeNet(l1=True),
        "model_l2": LeNet(l2=True),
        "model_l1_l2": LeNet(l1_l2=True),
        "model_svb": LeNet(svb=True),
        "model_soft_svb": LeNet(soft_svb=True),
        "model_jacobi_reg": LeNet(jacobi_reg=True),
        "model_jacobi_det_reg": LeNet(jacobi_det_reg=True),
        "model_dropout": LeNet(dropout_rate=0.5),
        "model_conf_penalty": LeNet(conf_penalty=True),
        "model_label_smoothing": LeNet(label_smoothing=True),
        "model_noise_inject_inputs": LeNet(noise_inject_inputs=True),
        "model_noise_inject_weights": LeNet(noise_inject_weights=True),
    }

    n_epochs = 100

    # Iterate through each model
    for model_name, model in models.items():
        print(f"Training model: {model_name} on dataset: {dataset}")

        # Check if there are multiple GPUs, and if so, use DataParallel
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = DataParallel(model)

        # Move the model to the device (CPU or GPU)
        model.to(device)

        losses, reg_losses, epochs, weights, train_accuracies, test_accuracies = train(
            train_loader, test_loader, model, device, n_epochs
        )

        torch.save(model.state_dict(), f"./trained_{dataset}_models/{model_name}.pt")

        # Save losses, reg_losses, epochs, weights, train_accuracies, test_accuracies using pickle
        data = {
            "losses": losses,
            "reg_losses": reg_losses,
            "epochs": epochs,
            "weights": weights,
            "train_accuracies": train_accuracies,
            "test_accuracies": test_accuracies,
        }

        with open(f"./trained_{dataset}_models/{model_name}_data.pkl", "wb") as f:
            pickle.dump(data, f)
