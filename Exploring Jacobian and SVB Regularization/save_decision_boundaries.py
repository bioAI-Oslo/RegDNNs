import jupyter_black
import torch
import numpy as np
from tqdm import tqdm

from data_generators import data_loader_MNIST, data_loader_CIFAR10, data_loader_CIFAR100
from plotting_tools import (
    plot_decision_boundaries_for_multiple_models,
    generate_random_vectors,
    get_random_img,
)
from tools import ModelInfo

jupyter_black.load()


if __name__ == "__main__":
    # Device configuration, use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader_CIFAR10, test_loader_CIFAR10 = data_loader_CIFAR10()

    # Set seeds for reproducibility
    seed_value = 42
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)

    # If you're using CUDA, also set this for reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True  # Necessary for reproducibility
        torch.backends.cudnn.benchmark = False  # Can affect performance

    model_names_set = [
        "model_no_reg",
        "model_no_reg_no_dropout",
        "model_l2",
        "model_jacobi",
        "model_jacobi_no_dropout",
        "model_svb",
    ]
    model_names = []

    for i in range(5):
        for name in model_names_set:
            model_names.append(f"{name}_{i}")
    models_CIFAR10 = {name: ModelInfo(name, "cifar10") for name in model_names}

    models_cifar10_with_names_0 = [
        (models_CIFAR10[f"model_no_reg_0"].model, "Only Dropout Regularization"),
        (models_CIFAR10[f"model_no_reg_no_dropout_0"].model, "No Regularization"),
        (models_CIFAR10[f"model_l2_0"].model, "L2 Regularization"),
        (models_CIFAR10[f"model_jacobi_0"].model, "Jacobian Regularization"),
        (
            models_CIFAR10[f"model_jacobi_no_dropout_0"].model,
            "Jacobian Regularization, No Dropout",
        ),
        (models_CIFAR10[f"model_svb_0"].model, "SVB Regularization"),
    ]
    models_cifar10_with_names_1 = [
        (models_CIFAR10[f"model_no_reg_1"].model, "Only Dropout Regularization"),
        (models_CIFAR10[f"model_no_reg_no_dropout_1"].model, "No Regularization"),
        (models_CIFAR10[f"model_l2_1"].model, "L2 Regularization"),
        (models_CIFAR10[f"model_jacobi_1"].model, "Jacobian Regularization"),
        (
            models_CIFAR10[f"model_jacobi_no_dropout_1"].model,
            "Jacobian Regularization, No Dropout",
        ),
        (models_CIFAR10[f"model_svb_1"].model, "SVB Regularization"),
    ]
    models_cifar10_with_names_2 = [
        (models_CIFAR10[f"model_no_reg_2"].model, "Only Dropout Regularization"),
        (models_CIFAR10[f"model_no_reg_no_dropout_2"].model, "No Regularization"),
        (models_CIFAR10[f"model_l2_2"].model, "L2 Regularization"),
        (models_CIFAR10[f"model_jacobi_2"].model, "Jacobian Regularization"),
        (
            models_CIFAR10[f"model_jacobi_no_dropout_2"].model,
            "Jacobian Regularization, No Dropout",
        ),
        (models_CIFAR10[f"model_svb_2"].model, "SVB Regularization"),
    ]
    models_cifar10_with_names_3 = [
        (models_CIFAR10[f"model_no_reg_3"].model, "Only Dropout Regularization"),
        (models_CIFAR10[f"model_no_reg_no_dropout_3"].model, "No Regularization"),
        (models_CIFAR10[f"model_l2_3"].model, "L2 Regularization"),
        (models_CIFAR10[f"model_jacobi_3"].model, "Jacobian Regularization"),
        (
            models_CIFAR10[f"model_jacobi_no_dropout_3"].model,
            "Jacobian Regularization, No Dropout",
        ),
        (models_CIFAR10[f"model_svb_3"].model, "SVB Regularization"),
    ]
    models_cifar10_with_names_4 = [
        (models_CIFAR10[f"model_no_reg_4"].model, "Only Dropout Regularization"),
        (models_CIFAR10[f"model_no_reg_no_dropout_4"].model, "No Regularization"),
        (models_CIFAR10[f"model_l2_4"].model, "L2 Regularization"),
        (models_CIFAR10[f"model_jacobi_4"].model, "Jacobian Regularization"),
        (
            models_CIFAR10[f"model_jacobi_no_dropout_4"].model,
            "Jacobian Regularization, No Dropout",
        ),
        (models_CIFAR10[f"model_svb_4"].model, "SVB Regularization"),
    ]

    image_0 = get_random_img(test_loader_CIFAR10)
    image_1 = get_random_img(test_loader_CIFAR10)
    image_2 = get_random_img(test_loader_CIFAR10)
    image_3 = get_random_img(test_loader_CIFAR10)
    image_4 = get_random_img(test_loader_CIFAR10)
    v1_0, v2_0 = generate_random_vectors(image_0)
    v1_1, v2_1 = generate_random_vectors(image_1)
    v1_2, v2_2 = generate_random_vectors(image_2)
    v1_3, v2_3 = generate_random_vectors(image_3)
    v1_4, v2_4 = generate_random_vectors(image_4)

    for i in tqdm(range(5)):
        plot_decision_boundaries_for_multiple_models(
            models_cifar10_with_names_0,
            "cifar10",
            image_0,
            v1_0,
            v2_0,
            device,
            resolution=250,
            # zoom = 0.01,
            zoom=0.0075,
            title="Decision Boundaries for DDNet Models Trained on CIFAR10",
            saveas=f"models_0_{i}",
        )
        plot_decision_boundaries_for_multiple_models(
            models_cifar10_with_names_1,
            "cifar10",
            image_1,
            v1_1,
            v2_1,
            device,
            resolution=250,
            # zoom = 0.01,
            zoom=0.0075,
            title="Decision Boundaries for DDNet Models Trained on CIFAR10",
            saveas=f"models_1_{i}",
        )
        plot_decision_boundaries_for_multiple_models(
            models_cifar10_with_names_2,
            "cifar10",
            image_2,
            v1_2,
            v2_2,
            device,
            resolution=250,
            # zoom = 0.01,
            zoom=0.0075,
            title="Decision Boundaries for DDNet Models Trained on CIFAR10",
            saveas=f"models_2_{i}",
        )
        plot_decision_boundaries_for_multiple_models(
            models_cifar10_with_names_3,
            "cifar10",
            image_3,
            v1_3,
            v2_3,
            device,
            resolution=250,
            # zoom = 0.01,
            zoom=0.0075,
            title="Decision Boundaries for DDNet Models Trained on CIFAR10",
            saveas=f"models_3_{i}",
        )
        plot_decision_boundaries_for_multiple_models(
            models_cifar10_with_names_4,
            "cifar10",
            image_4,
            v1_4,
            v2_4,
            device,
            resolution=250,
            # zoom = 0.01,
            zoom=0.0075,
            title="Decision Boundaries for DDNet Models Trained on CIFAR10",
            saveas=f"models_4_{i}",
        )
