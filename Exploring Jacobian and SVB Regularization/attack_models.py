import torch
import pickle
from data_generators import data_loader_MNIST, data_loader_CIFAR10, data_loader_CIFAR100
from tools import ModelInfo
from attack_tools import fgsm_attack_test, pgd_attack_test

if __name__ == "__main__":
    # Device configuration, use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set dataset
    dataset = "mnist"
    # dataset = "cifar10"
    # dataset = "cifar100"

    if dataset == "mnist":
        _, test_loader = data_loader_MNIST()
    elif dataset == "cifar10":
        _, test_loader = data_loader_CIFAR10()
    elif dataset == "cifar100":
        _, test_loader = data_loader_CIFAR100()

    # Define the models
    model_names = [
        "model_no_reg_0",
        "model_no_reg_no_dropout_0",
        "model_l2_0",
        "model_l2_no_dropout_0",
        "model_jacobi_0",
        "model_jacobi_no_dropout_0",
        "model_svb_0",
        "model_svb_no_dropout_0",
        "model_jacobi_and_svb_0",
        "model_all_0",
        "model_jacobi_and_l2_0",
        "model_svb_and_l2_0",
    ]
    models = {name: ModelInfo(name, dataset) for name in model_names}

    # Load the trained models
    for model_name in model_names:
        models[model_name].model.load_state_dict(
            torch.load(f"./trained_{dataset}_models/{model_name}.pt")
        )
        models[model_name].model.to(device)
        models[model_name].model.eval()

    # Epsilons to use for FGSM attacks
    epsilons = [
        0,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
    ]

    # Perform FGSM attack and store results
    for model_name, model in models.items():
        print(f"Attacking model: {model_name} on dataset: {dataset} with FGSM")

        # Store accuracies for all epsilons
        accuracies = []

        # Perform the attack for each epsilon
        for eps in epsilons:
            acc = fgsm_attack_test(model, device, test_loader, eps)
            accuracies.append(acc)

        # Save the accuracies for all epsilons for this model
        with open(
            f"./attacked_{dataset}_models/{model_name}_fgsm_accuracies.pkl", "wb"
        ) as f:
            pickle.dump({"epsilons": epsilons, "accuracies": accuracies}, f)
