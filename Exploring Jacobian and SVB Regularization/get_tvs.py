import torch
import numpy as np
import pandas as pd
import pickle
from scipy import stats
from tqdm import tqdm
from data_generators import data_loader_MNIST, data_loader_CIFAR10, data_loader_CIFAR100
from tools import ModelInfo, compute_total_variation
from plotting_tools import get_random_img, generate_random_vectors

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
        model = models[name].model  # Get model
        tv_values = {
            zoom: [] for zoom in zoom_levels
        }  # To store total variation values for each zoom level

        # Generate tv values for n_images number of images
        for _ in range(n_images):
            # Get random image and vectors
            img = get_random_img(test_loader)
            v1, v2 = generate_random_vectors(img)

            img = img.to(device)

            # Compute the isotropic total variation of decision boundaries for the current model
            # and the generated image at different zoom levels
            tv_list = compute_total_variation(
                model,
                img,
                v1,
                v2,
                device,
                resolution=300,
                zoom=zoom_levels,
                mode=mode,
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
