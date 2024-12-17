import json
import pickle
import argparse
import random
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from pip._vendor import tomli as tomllib

import torch
import torchvision.transforms as transforms
import torchvision

from predcoding.snn.network import EnergySNN
from predcoding.experiments.eval import test_reconstruction
from predcoding.experiments.mismatch import get_mismatch_results, plot_voltage_diff
from predcoding.utils import model_result_dict_load


def main():
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(42)
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--config-file")
    args = parser.parse_args()

    with open(args.config_file, "rb") as f:
        config = tomllib.load(f)
    print(json.dumps(config, indent=2))

    # timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")
    # file_name = config["checkpoint"]["file_name"]

    # network parameters
    d_in = config["network"]["d_in"]
    d_hidden = config["network"]["d_hidden"]
    d_out = config["network"]["d_out"]

    # training parameters
    T = config["training"]["T"]

    # data loaders
    batch_size = config["training"]["n_batch"]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    testdata = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testdata, batch_size=batch_size, shuffle=False, num_workers=2)

    # define network
    model_E = EnergySNN(
        d_in,
        d_hidden,
        d_out=d_out,
        is_adaptive=config["network"]["use_alif_neurons"],
        one_to_one=config["network"]["one_to_one"],
        p_dropout=config["network"]["p_dropout"],
        is_recurrent=config["network"]["is_recurrent"],
        b0=config["network"]["b0"],
        device=device,
    ).to(device)
    model_NE = EnergySNN(
        d_in,
        d_hidden,
        d_out=d_out,
        is_adaptive=config["network"]["use_alif_neurons"],
        one_to_one=config["network"]["one_to_one"],
        p_dropout=config["network"]["p_dropout"],
        is_recurrent=config["network"]["is_recurrent"],
        b0=config["network"]["b0"],
        device=device,
    ).to(device)

    saved_dict = model_result_dict_load("./checkpoints/bottleneck_E/best_model_bottleneck_E.pt.tar")
    model_E.load_state_dict(saved_dict["state_dict"])
    saved_dict = model_result_dict_load("./checkpoints/bottleneck_NE/best_model_bottleneck_NE.pt.tar")
    model_NE.load_state_dict(saved_dict["state_dict"])

    model_E.eval()
    model_NE.eval()

    test_recon_loss = test_reconstruction(model_E, test_loader, T)
    print(f"Energy test recon loss: {test_recon_loss:.2f}")
    test_recon_loss = test_reconstruction(model_NE, test_loader, T)
    print(f"Control test recon loss: {test_recon_loss:.2f}")

    # Mismatch experiments
    keep_time = False  # value of each neuron is averaged over all timepoints
    element = "apical"
    l_clamp_expected_E, l_clamp_surprise_E = get_mismatch_results(model_E, test_loader, T, element, keep_time)
    l_clamp_diff_E = [(l_e - l_s).mean(dim=1).cpu().numpy() for l_e, l_s in zip(l_clamp_expected_E, l_clamp_surprise_E)]

    l_clamp_expected_NE, l_clamp_surprise_NE = get_mismatch_results(model_NE, test_loader, T, element, keep_time)
    l_clamp_diff_NE = [
        (l_e - l_s).mean(dim=1).cpu().numpy() for l_e, l_s in zip(l_clamp_expected_NE, l_clamp_surprise_NE)
    ]

    with open("./checkpoints/mismatch/l_clamp_diff_E.p", "wb") as f:
        pickle.dump(l_clamp_diff_E, f)

    with open("./checkpoints/mismatch/l_clamp_diff_NE.p", "wb") as f:
        pickle.dump(l_clamp_diff_NE, f)

    with open("./checkpoints/mismatch/l_clamp_diff_E.p", "rb") as f:
        l_clamp_diff_E = pickle.load(f)

    with open("./checkpoints/mismatch/l_clamp_diff_NE.p", "rb") as f:
        l_clamp_diff_NE = pickle.load(f)

    plot_voltage_diff(*l_clamp_diff_NE, *l_clamp_diff_E)


if __name__ == "__main__":
    main()
