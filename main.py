import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from predcoding.snn.network import SnnNetwork3Layer
from predcoding.training import train_fptt, get_stats_named_params, reset_named_params
from predcoding.experiments.eval import test
from predcoding.experiments.decoder import train_linear_proj
from predcoding.utils import count_parameters, save_checkpoint

def main():
    # set seed
    torch.manual_seed(999)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])

    batch_size = 200

    traindata = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    testdata = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # data loading
    train_loader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testdata, batch_size=batch_size, shuffle=False, num_workers=2)

    # network parameters
    adap_neuron = True  # whether use adaptive neuron or not
    clf_alpha = 1
    energy_alpha = 0.05  # - config.clf_alpha
    spike_alpha = 0.0  # energy loss on spikes
    num_readout = 10
    onetoone = True
    lr = 1e-3
    alg = "fptt"
    dp = 0.4
    is_rec = False
    b_j0 = 0.1      # neural threshold baseline
    R_m = 3         # membrane resistance
    gamma = 0.5     # gradient scale
    lens = 0.5
    baseline_threshold = b_j0

    # training parameters
    T = 50
    K = 10  # k_updates is num updates per sequence
    omega = int(T / K)  # update frequency
    clip = 1.0
    log_interval = 20
    epochs = 35
    alpha = 0.2
    beta = 0.5
    rho = 0.0 

    # set input and t param
    IN_dim = 784
    hidden_dim = [600, 500, 500]
    n_classes = 10

    # define network
    model = SnnNetwork3Layer(
        IN_dim,
        hidden_dim,
        n_classes,
        is_adapt=adap_neuron,
        one_to_one=onetoone,
        dp_rate=dp,
        is_rec=is_rec,
        b_j0=b_j0,
        device=device
    )
    model.to(device)
    # print(model)

    # define new loss and optimiser
    total_params = count_parameters(model)
    print("total param count %i" % total_params)

    # define optimiser
    optimizer = optim.Adamax(model.parameters(), lr=lr, weight_decay=0.0001)
    # reduce the learning after 20 epochs by a factor of 10
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    # untrained network
    test_loss, acc1 = test(model, test_loader, T)

    named_params = get_stats_named_params(model)
    all_test_losses = []
    best_acc1 = 0

    for epoch in range(epochs):
        train_fptt(
            epoch,
            batch_size,
            log_interval,
            train_loader,
            model,
            named_params,
            T,
            K,
            omega,
            optimizer,
            clf_alpha,
            energy_alpha,
            spike_alpha,
            clip,
            lr,
            alpha,
            beta,
            rho
        )

        reset_named_params(named_params)

        test_loss, acc1 = test(model, test_loader, T)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if is_best:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "optimizer": optimizer.state_dict(),
                },
                prefix="checkpoints/",
                filename="best_model.pt.tar",
            )

        all_test_losses.append(test_loss)

    model.eval()
    test(model, test_loader, T)



if __name__ == "main":
    main()
