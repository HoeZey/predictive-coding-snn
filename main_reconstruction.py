import json
from pip._vendor import tomli as tomllib
import argparse

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from predcoding.snn.network import EnergySNN
from predcoding.experiments.eval import test
from predcoding.experiments.decoder import train_linear_proj, get_states
from predcoding.utils import model_result_dict_load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--config-file")
    args = parser.parse_args()

    with open(args.config_file, "rb") as f:
        config = tomllib.load(f)
    print(json.dumps(config, indent=2))

    # network parameters
    d_in = config["network"]["d_in"]
    d_hidden = config["network"]["d_hidden"]
    n_classes = config["network"]["n_classes"]
    T = config["training"]["T"]
    file_name = config["checkpoint"]["file_name"]

    # device
    torch.manual_seed(999)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # data loaders
    batch_size = config["training"]["n_batch"]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    testdata = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testdata, batch_size=batch_size, shuffle=False, num_workers=2)

    # define network
    model = EnergySNN(
        d_in,
        d_hidden,
        d_out=n_classes,
        is_adaptive=config["network"]["use_alif_neurons"],
        one_to_one=config["network"]["one_to_one"],
        p_dropout=config["network"]["p_dropout"],
        is_recurrent=config["network"]["is_recurrent"],
        b0=config["network"]["b0"],
        device=device,
    ).to(device)

    saved_dict = model_result_dict_load(f"./checkpoints/best_model_{file_name}.pt.tar")
    model.load_state_dict(saved_dict["state_dict"])

    model.eval()
    # test(model, test_loader, T)

    # get params and put into dict
    param_names_wE = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_names_wE.append(name)

    # clamped generation of internal representations
    no_input = torch.zeros((1, d_in)).to(device)
    clamp_T = T * 5

    layers = config["reconstruction"]["decoder_layers"]
    n_layers = len(layers)

    clamp_E_all = [np.zeros((10, d_hidden[layer])) for layer in range(n_layers)]

    for l in range(n_layers):
        for nr in range(n_classes):
            with torch.no_grad():
                model.eval()
                _, hidden_gen_E_ = model.clamped_generate(nr, no_input, *model.init_hidden(1), clamp_T, clamp_value=1)

            l_E = get_states([hidden_gen_E_], l, d_hidden[l], batch_size=1, T=clamp_T)

            clamp_E_all[l][nr] += l_E.mean(axis=1).squeeze().cpu().numpy()

        torch.cuda.empty_cache()

    ##############################################################
    # decode from clamped representations
    ##############################################################
    fn_loss = nn.MSELoss()
    test_loader2 = torch.utils.data.DataLoader(testdata, batch_size=batch_size, shuffle=False, num_workers=2)

    print("TRAINING DECODERS")
    decoders = [
        (
            layer,
            train_linear_proj(20, layer, model, test_loader2, d_hidden[layer], d_in, T, device, fn_loss)[0],
        )
        for layer in layers
    ]

    print("GENERATING PLOT")
    # plot decoding of clamped internal representations
    fig, axes = plt.subplots(len(decoders), 10, figsize=(20, 2 * n_layers), dpi=200)
    axes = np.flip(axes.reshape(n_layers, n_classes), axis=0)

    with torch.no_grad():
        for l, ((layer, decoder), clamp_E) in enumerate(zip(decoders, clamp_E_all)):
            decoder.eval()
            axes[l, 0].set_ylabel(f"L{l + 1}")
            for nr in range(n_classes):
                img = (
                    decoder(torch.tensor(clamp_E[nr].astype("float32")).to(device).view(-1, d_hidden[layer]))
                    .reshape(28, 28)
                    .cpu()
                )
                axes[l, nr].imshow(img, cmap="viridis")
                axes[l, nr].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

    for nr in range(n_classes):
        axes[-1, nr].set_title(str(nr))

    fig.suptitle(f"Reconstructions from clamps reps back to image plane")

    plt.tight_layout()
    plt.savefig(f"images/reconstruction_{file_name}.png")


if __name__ == "__main__":
    main()
