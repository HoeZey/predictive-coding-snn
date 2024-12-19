from pathlib import Path
import json
from pip._vendor import tomli as tomllib
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torchvision.transforms as transforms
import torchvision

from predcoding.snn.network import EnergySNN
from predcoding.experiments.eval import test_reconstruction
from predcoding.experiments.decoder import train_clf_decoder
from predcoding.utils import model_result_dict_load


def main():
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--config-file")
    args = parser.parse_args()

    with open(args.config_file, "rb") as f:
        config = tomllib.load(f)
    print(json.dumps(config, indent=2))

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")
    file_name = config["checkpoint"]["file_name"]

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
    model = EnergySNN(
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

    saved_dict = model_result_dict_load(f"./checkpoints/{file_name}/best_model_{file_name}.pt.tar")
    model.load_state_dict(saved_dict["state_dict"])

    model.eval()
    test_recon_loss = test_reconstruction(model, test_loader, T)
    print(f"Test recon loss: {test_recon_loss:.2f}")

    fig_loss, axs_loss = plt.subplots(3, 1, figsize=(7, 4), dpi=200, sharex=True)
    fig_acc, axs_acc = plt.subplots(3, 1, figsize=(7, 4), dpi=200, sharex=True)
    axs_loss = axs_loss.flatten()
    axs_acc = axs_acc.flatten()

    for layer in config["reconstruction"]["decoder_layers"]:
        _, losses, accs = train_clf_decoder(
            model=model,
            layer_to_decode=layer,
            d_hidden=d_hidden[layer],
            data_loader=test_loader,
            epochs=100,
            T=T,
            device=device,
        )
        axs_loss[layer].plot(range(len(losses)), losses)
        axs_acc[layer].plot(range(len(accs)), accs)

    test_losses_img_path = Path(f"./images/clf_losses/{file_name}/{timestamp}")
    test_accs_img_path = Path(f"./images/clf_accs/{file_name}/{timestamp}")
    test_losses_img_path.mkdir(parents=True, exist_ok=True)
    test_accs_img_path.mkdir(parents=True, exist_ok=True)

    fig_loss.savefig(f"{test_losses_img_path}/losses.png")
    fig_acc.savefig(f"{test_accs_img_path}/accs.png")


if __name__ == "__main__":
    main()
