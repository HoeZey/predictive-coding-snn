import json
from pip._vendor import tomli as tomllib
import argparse

import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm

from predcoding.snn.network import EnergySNN
from predcoding.training import train_fptt_bottleneck, get_stats_named_params, reset_named_params
from predcoding.experiments.eval import test
from predcoding.experiments.decoder import LinearDecoder
from predcoding.utils import count_parameters, save_checkpoint


def main():
    # set seed
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--config-file")
    args = parser.parse_args()

    with open(args.config_file, "rb") as f:
        config = tomllib.load(f)
    print(json.dumps(config, indent=2))

    # network parameters
    d_in = config["network"]["d_in"]
    d_hidden = config["network"]["d_hidden"]
    d_out = config["network"]["d_out"]

    # training parameters
    epochs = config["training"]["epochs"]
    lr = config["training"]["lr"]
    T = config["training"]["T"]
    K = config["training"]["K"]  # k_updates is num updates per sequence
    omega = int(T / K)  # update frequency

    # device
    torch.manual_seed(999)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # data loaders
    batch_size = config["training"]["n_batch"]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    traindata = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    testdata = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=False, num_workers=2)

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

    model_param_count = count_parameters(model)
    print(f"Model params: {model_param_count}")

    # define optimiser
    optimizer = optim.Adamax(model.parameters(), lr=lr, weight_decay=0.0001)
    model_params = get_stats_named_params(model)

    model.train()

    for epoch in tqdm(range(epochs), total=epochs):
        train_fptt_bottleneck(
            # models
            model=model,
            # training
            T=T,  # number of timesteps
            K=K,  # number of updates
            update_interval=omega,
            data_loader=train_loader,
            model_params=model_params,
            # optimization
            optimizer=optimizer,
            alpha_recon=config["decoder"]["recon_alpha"],
            alpha_energy=config["training"]["energy_alpha"],
            clip_value=config["training"]["clip"],
            # fptt regularizer parameters
            alpha=config["training"]["alpha"],
            beta=config["training"]["beta"],
            rho=config["training"]["rho"],
            # other
            epoch=epoch,
            log_interval=config["training"]["log_interval"],
            debug=False,
        )

        reset_named_params(model_params)

    save_checkpoint(
        {"state_dict": model.to("cpu").state_dict()},
        prefix="checkpoints/train/",
        filename=f"best_model_{config['checkpoint']['file_name']}.pt.tar",
    )


if __name__ == "__main__":
    main()
