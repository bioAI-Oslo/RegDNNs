import torch
import pickle
from data_generators import data_loader_MNIST, data_loader_CIFAR10, data_loader_CIFAR100
from tools import ModelInfo
from attack_tools import fgsm_attack_test, pgd_attack_test

if __name__ == "__main__":
    # Device configuration, use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set dataset
    # dataset = "mnist"
    # dataset = "cifar10"
    dataset = "cifar100"

    # Set attack
    # attack = "fgsm"
    attack = "pgd"

    if dataset == "mnist":
        _, test_loader = data_loader_MNIST()
    elif dataset == "cifar10":
        _, test_loader = data_loader_CIFAR10()
    elif dataset == "cifar100":
        _, test_loader = data_loader_CIFAR100()

    # Define the models
    model_names = [
        "model_no_reg_0",
        "model_l2_0",
        "model_jacobi_0",
        "model_jacobi_no_dropout_0",
        "model_svb_0",
        "model_no_reg_1",
        "model_l2_1",
        "model_jacobi_1",
        "model_jacobi_no_dropout_1",
        "model_svb_1",
        "model_no_reg_2",
        "model_l2_2",
        "model_jacobi_2",
        "model_jacobi_no_dropout_2",
        "model_svb_2",
        "model_no_reg_3",
        "model_l2_3",
        "model_jacobi_3",
        "model_jacobi_no_dropout_3",
        "model_svb_3",
        "model_no_reg_4",
        "model_l2_4",
        "model_jacobi_4",
        "model_jacobi_no_dropout_4",
        "model_svb_4",
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
        # 0.35,
        # 0.4,
        # 0.45,
        # 0.5,
        # 0.55,
        # 0.6,
        # 0.65,
        # 0.7,
        # 0.75,
    ]

    # Iterations to use for PGD attack
    iters_list = [0, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30]

    # Perform attack and store results
    for model_name in model_names:
        print(f"Attacking model: {model_name} on dataset: {dataset}")

        # Store accuracies for all epsilons/iterations
        accuracies = []

        # Perform the attack for each epsilon/iteration
        if attack == "pgd":
            for iter in iters_list:
                acc = pgd_attack_test(
                    models[model_name].model,
                    device,
                    test_loader,
                    eps=32
                    / 255,  # Four times the value from Hoffman 2019 (which was 32/255)
                    alpha=1
                    / 255,  # Four times the value from Hoffman 2019 (which was 1/255)
                    iters=iter,
                    dataset=dataset,
                )
                accuracies.append(acc)
        elif attack == "fgsm":
            for eps in epsilons:
                acc = fgsm_attack_test(
                    models[model_name].model,
                    device,
                    test_loader,
                    epsilon=eps,
                    dataset=dataset,
                )
                accuracies.append(acc)

        # Save the accuracies for all epsilons for this model
        with open(
            f"./attacked_{dataset}_models/{model_name}_{attack}_accuracies.pkl", "wb"
        ) as f:
            if attack == "fgsm":
                pickle.dump({"epsilons": epsilons, "accuracies": accuracies}, f)
            elif attack == "pgd":
                pickle.dump({"iters_list": iters_list, "accuracies": accuracies}, f)
