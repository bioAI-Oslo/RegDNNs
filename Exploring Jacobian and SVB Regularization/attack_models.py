"""
This script performs adversarial attacks on different neural network models trained on datasets MNIST, CIFAR-10, and CIFAR-100. 
Two types of attacks are supported: Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD).

Structure:
1. Set device configuration (use GPU if available).
2. Set the dataset (MNIST, CIFAR-10, or CIFAR-100) to be used.
3. Set the type of attack to be performed (FGSM or PGD).
4. Load the appropriate test data based on the chosen dataset.
5. Define the names of models to be attacked.
6. Set the epsilon values for FGSM attacks and the iteration values for PGD attacks.
7. For each model:
    - Load the model.
    - For each epsilon (for FGSM) or iteration (for PGD), perform the attack and record the accuracy.
    - Save the recorded accuracies to a pickle file.

Output:
The output accuracies of the models post-attack are saved in pickle files under the `./attacked_{dataset}_models/` directory.
"""

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
    dataset = "cifar100"  # Cifar100 does not have models with or without dropout

    # Set attack
    attack = "fgsm"
    # attack = "pgd"

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
        "model_jacobi_0",
        "model_jacobi_no_dropout_0",
        "model_svb_0",
    ]
    models = {name: ModelInfo(name, dataset) for name in model_names}

    # Epsilons to use for FGSM attacks
    epsilons = [
        0,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
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
                    eps=32 / 255,  # The value from Hoffman 2019 is 32/255
                    alpha=1 / 255,  # The value from Hoffman 2019 is 1/255
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
