import json
from pip._vendor import tomli as tomllib
import argparse

import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm

from predcoding.snn.network import EnergySNN
from predcoding.training import train_fptt, get_stats_named_params, reset_named_params
from predcoding.experiments.eval import test
from predcoding.experiments.decoder import LinearReadout
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
    n_classes = config["network"]["n_classes"]

    # training parameters
    supervised = config["training"]["supervised"]
    epochs = config["training"]["epochs"]
    lr = config["training"]["lr"]
    T = config["training"]["T"]
    K = config["training"]["K"]  # k_updates is num updates per sequence
    omega = int(T / K)  # update frequency

    # self_supervised params
    self_supervised = config["decoder"]["self_supervised"]
    recon_alpha = config["decoder"]["recon_alpha"]
    decoder_layer = config["decoder"]["decoder_layer"]

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

    decoder = LinearReadout(d_in=d_hidden[decoder_layer], d_out=d_in, device=device).to(device)

    model_params = count_parameters(model)
    decoder_params = count_parameters(decoder)
    print(f"Model params: {model_params} | Decoder params: {decoder_params} | Tota: {model_params + decoder_params}")

    # define optimiser
    optimizer = optim.Adamax(model.parameters(), lr=lr, weight_decay=0.0001)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr, weight_decay=0.0001)

    # reduce the learning after 20 epochs by a factor of 10
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    named_params = get_stats_named_params(model)
    all_test_losses = []
    best_acc1 = 0

    model.train()
    decoder.train()

    for epoch in tqdm(range(epochs), total=epochs):
        train_fptt(
            epoch=epoch,
            batch_size=batch_size,
            log_interval=config["training"]["log_interval"],
            train_loader=train_loader,
            model=model,
            named_params=named_params,
            time_steps=T,
            k_updates=K,
            omega=omega,
            optimizer=optimizer,
            clf_alpha=config["training"]["clf_alpha"],
            energy_alpha=config["training"]["energy_alpha"],
            spike_alpha=config["training"]["spike_alpha"],
            clip=config["training"]["clip"],
            alpha=config["training"]["alpha"],
            beta=config["training"]["beta"],
            rho=config["training"]["rho"],
            # self-supervised params
            supervised=supervised,
            self_supervised=self_supervised,
            decoder=decoder,
            decoder_layer=decoder_layer,
            recon_alpha=recon_alpha,
            decoder_optimizer=decoder_optimizer,
        )

        reset_named_params(named_params)

        test_loss, acc1 = test(model, test_loader, T)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if is_best:
            save_checkpoint(
                {"state_dict": model.to("cpu").state_dict()},
                prefix="checkpoints/",
                filename=f"best_model_{config['checkpoint']['file_name']}.pt.tar",
            )
            save_checkpoint(
                {"state_dict": decoder.to("cpu").state_dict()},
                prefix="checkpoints/",
                filename=f"decoder_{config['checkpoint']['file_name']}.pt.tar",
            )

            model.to(device)

        all_test_losses.append(test_loss)

    model.eval()
    test(model, test_loader, T)


if __name__ == "__main__":
    main()
