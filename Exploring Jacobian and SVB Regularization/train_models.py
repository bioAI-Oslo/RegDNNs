import torch
import os
import pickle
from data_generators import data_loader_MNIST
from model_classes import LeNet_MNIST
from tools import train
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
    svb_freq = 600
    svb_eps = 0.05

    # Initialize model
    model = LeNet_MNIST(dropout_rate=0.0, l2_lmbd=0.0)

    # Check if there are multiple GPUs, and if so, use DataParallel
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = DataParallel(model)

    # Move the model to the device (CPU or GPU)
    model = model.to(device)

    n_epochs = 250  # 250 in Hoffman 2019
    (
        losses,
        reg_losses,
        epochs,
        train_accuracies,
        test_accuracies,
    ) = train(train_loader, test_loader, model, device, n_epochs)

    # Switch to evaluation mode
    model.eval()

    if not os.path.exists("./trained_models"):
        os.makedirs("./trained_models")

    # If the model was trained with DataParallel, save model.module.state_dict().
    # Otherwise, just save model.state_dict()
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), "./trained_models/model_no_reg_no_dropout.pt")
    else:
        torch.save(model.state_dict(), "./trained_models/model_no_reg_no_dropout.pt")

    # Save losses, reg_losses, epochs, train_accuracies, test_accuracies using pickle
    data = {
        "losses": losses,
        "reg_losses": reg_losses,
        "epochs": epochs,
        "train_accuracies": train_accuracies,
        "test_accuracies": test_accuracies,
    }

    with open("./trained_models/model_no_reg_no_dropout_data.pkl", "wb") as f:
        pickle.dump(data, f)
