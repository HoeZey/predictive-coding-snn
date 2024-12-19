import random
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader

from predcoding.snn.network import EnergySNN
from predcoding.experiments.decoder import get_states


def get_clamped_reps(model: EnergySNN, data_loader: DataLoader, T: int, element: str, keep_time: bool, expected: bool):
    zero_input = torch.zeros((1, model.d_out), device=model.device)

    n_data = len(data_loader.dataset)
    if keep_time:
        l_clamp_layers = [torch.zeros((n_data, int(T / 2), d)) for d in model.d_hidden]
    else:
        l_clamp_layers = [torch.zeros((n_data, d)) for d in model.d_hidden]

    i_data = 0
    model.eval()
    for _, (data, labels) in tqdm(enumerate(data_loader)):
        # For each batch go through each number and calculate clamped representation for that number
        for nr in range(10):
            where_is_number = labels == nr
            number_without_current = list(set(range(10)) - set([nr]))
            for i_img in range(where_is_number.sum()):
                image_input = data[where_is_number][i_img].reshape(-1, model.d_out).to(model.device)
                if expected:
                    readout_clamped = image_input
                else:
                    surprise_class = random.choice(number_without_current)
                    readout_clamped = data[labels == surprise_class][0].reshape(-1, model.d_out).to(model.device)

                with torch.no_grad():
                    h, readout = model.init_hidden(1)

                    # no input for T/4 timesteps before stimulus onset
                    # clamped stimulus for T/2 timesteps
                    # no input for T/4 timesteps after stimulus
                    h, readout = model.inference(zero_input, h, readout, int(T / 4), clamp=False)
                    h_hist, readout = model.inference(image_input, h[-1], readout_clamped, int(T / 2), clamp=True)
                    h, readout = model.inference(zero_input, h[-1], readout, int(T / 4), clamp=False)

                    l_layers = [
                        get_states(h_hist, l, d_hidden, batch_size=1, T=int(T / frac), element=element)
                        for l, (d_hidden, frac) in enumerate(zip(model.d_hidden, [4, 2, 4]))
                    ]

                    for l_clamp, l_i in zip(l_clamp_layers, l_layers):
                        l_clamp[i_data] += (l_i if keep_time else l_i.mean(dim=1)).squeeze()

                    i_data += 1
                torch.cuda.empty_cache()

    return l_clamp_layers


def get_mismatch_results(model: EnergySNN, data_loader: DataLoader, T: int, element: str, keep_time: bool):
    torch.manual_seed(42)
    random.seed(42)
    l_clamp_expected = get_clamped_reps(model, data_loader, T, element, keep_time, expected=True)
    torch.manual_seed(42)
    random.seed(42)
    l_clamp_surprise = get_clamped_reps(model, data_loader, T, element, keep_time, expected=False)

    return l_clamp_expected, l_clamp_surprise


def plot_voltage_diff(
    l1_diff_control, l2_diff_control, l3_diff_control, l1_diff_energy, l2_diff_energy, l3_diff_energy
):
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    _, bins1, _ = ax[0].hist(
        l1_diff_control.flatten(),
        weights=[100 / len(l1_diff_control.flatten())] * len(l1_diff_control.flatten()),
        bins=30,
        log=True,
    )
    _, bins2, _ = ax[1].hist(
        l2_diff_control.flatten(),
        weights=[100 / len(l2_diff_control.flatten())] * len(l2_diff_control.flatten()),
        bins=30,
        log=True,
        label="Control",
    )
    _, bins3, _ = ax[2].hist(
        l3_diff_control.flatten(),
        weights=[100 / len(l3_diff_control.flatten())] * len(l3_diff_control.flatten()),
        bins=30,
        log=True,
    )

    ax[0].hist(
        l1_diff_energy.flatten(),
        weights=[100 / len(l1_diff_energy.flatten())] * len(l1_diff_energy.flatten()),
        bins=bins1,
        alpha=0.5,
        log=True,
    )
    ax[1].hist(
        l2_diff_energy.flatten(),
        weights=[100 / len(l2_diff_energy.flatten())] * len(l2_diff_energy.flatten()),
        bins=bins2,
        alpha=0.5,
        log=True,
        label="Energy",
    )
    ax[2].hist(
        l3_diff_energy.flatten(),
        weights=[100 / len(l3_diff_energy.flatten())] * len(l3_diff_energy.flatten()),
        bins=bins3,
        alpha=0.5,
        log=True,
    )

    ax[1].legend()

    ax[0].xaxis.set_label_position("top")
    ax[0].xaxis.set_label_position("top")

    ax[0].set_ylabel("L1\nPercentage")
    ax[1].set_ylabel("L2\nPercentage")
    ax[2].set_ylabel("L3\nPercentage")
    fig.savefig("test.png")
