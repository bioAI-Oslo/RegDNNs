import torch
import numpy as np
import pandas as pd
import pickle
from scipy import stats
from tqdm import tqdm
from data_generators import data_loader_MNIST, data_loader_CIFAR10, data_loader_CIFAR100
from tools import ModelInfo, compute_total_variation
from plotting_tools import get_random_img, generate_random_vectors


# Setting random seeds for reproducibility
SEED_VALUE = 42

import random

random.seed(SEED_VALUE)

np.random.seed(SEED_VALUE)

torch.manual_seed(SEED_VALUE)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED_VALUE)
    torch.cuda.manual_seed_all(SEED_VALUE)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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

    model_names_set = [
        "model_no_reg",
        "model_no_reg_no_dropout",
        "model_l2",
        "model_jacobi",
        "model_jacobi_no_dropout",  # Not included for ResNet18
        "model_svb",
    ]
    model_names = []

    for i in range(5):
        for name in model_names_set:
            model_names.append(f"{name}_{i}")

    models = {name: ModelInfo(name, dataset) for name in model_names}

    # Define zoom levels, number of images, and confidence level for confidence interval calculation
    zoom_levels = [0.025, 0.01, 0.001]
    n_images = 50
    confidence_level = 0.95
    mode = "anisotropic"

    # Dataframe to store results
    cols = pd.MultiIndex.from_product([zoom_levels, ["mean", "conf_interval"]])
    df_results = pd.DataFrame(index=model_names, columns=cols)

    # Loop over the selected models
    for name in tqdm(model_names):
        model = models[name].model.to(device)  # Get model
        tv_values = {
            zoom: [] for zoom in zoom_levels
        }  # To store total variation values for each zoom level

        # Generate tv values for n_images number of images
        for _ in range(n_images):
            # Get random image and vectors
            img = get_random_img(test_loader)
            v1, v2 = generate_random_vectors(img)

            img = img.to(device)
            v1 = v1.to(device)
            v2 = v2.to(device)

            # Compute the isotropic total variation of decision boundaries for the current model
            # and the generated image at different zoom levels
            tv_list = compute_total_variation(
                model,
                img,
                v1,
                v2,
                device,
                resolution=250,
                zoom=zoom_levels,
                mode=mode,
                dataset=dataset,
            )
            for zoom, tv in zip(zoom_levels, tv_list):
                tv_values[zoom].append(tv)

        # Calculate mean and 95% confidence interval for total variation values at each zoom level
        for zoom in zoom_levels:
            mean = np.mean(tv_values[zoom])
            std_err = np.std(tv_values[zoom]) / np.sqrt(n_images)
            conf_interval = (
                stats.t.ppf((1 + confidence_level) / 2, n_images - 1) * std_err
            )
            df_results.loc[name, (zoom, "mean")] = mean
            df_results.loc[name, (zoom, "conf_interval")] = conf_interval

        # Save the results
        with open(
            f"./total_variation_{dataset}_models/{name}_total_{mode}_variation.pkl",
            "wb",
        ) as f:
            pickle.dump(df_results, f)
