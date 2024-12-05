import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm

from predcoding.snn.network import EnergySNN
from predcoding.training import train_fptt, get_stats_named_params, reset_named_params
from predcoding.experiments.eval import test
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
    use_alif_neurons = True  # whether use adaptive neuron or not
    clf_alpha = 1
    energy_alpha = 0.05  # - config.clf_alpha
    spike_alpha = 0.0  # energy loss on spikes
    one_to_one = True
    lr = 1e-3
    p_dropout = 0.4
    is_recurrent = False
    b0 = 0.1  # neural threshold baseline

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
    # set input and t param
    d_in = 784
    d_hidden = [600, 500, 500]
    n_classes = 10

    # define network
    model = EnergySNN(
        d_in,
        d_hidden,
        d_out=n_classes,
        is_adaptive=use_alif_neurons,
        one_to_one=one_to_one,
        p_dropout=p_dropout,
        is_recurrent=is_recurrent,
        b0=b0,
        device=device,
    )
    model.to(device)
    # print(model)

    # define new loss and optimiser
    total_params = count_parameters(model)
    print(f"Total param count {total_params}")

    # define optimiser
    optimizer = optim.Adamax(model.parameters(), lr=lr, weight_decay=0.0001)
    # reduce the learning after 20 epochs by a factor of 10
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    named_params = get_stats_named_params(model)
    all_test_losses = []
    best_acc1 = 0

    model.train()
    for epoch in tqdm(range(epochs), total=epochs):
        print(f"Epoch {epoch}: ", end="")
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
            rho,
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
                    "state_dict": model.detach().cpu().state_dict(),
                    "best_acc1": best_acc1,
                    "optimizer": optimizer.state_dict(),
                },
                prefix="checkpoints/",
                filename="best_model.pt.tar",
            )

        all_test_losses.append(test_loss)

    model.eval()
    test(model, test_loader, T)


if __name__ == "__main__":
    main()
