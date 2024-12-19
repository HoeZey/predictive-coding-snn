import torch
import torch.nn.functional as F
from predcoding.snn.network import EnergySNN
from predcoding.experiments.decoder import LinearDecoder, get_states
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def get_stats_named_params(model: EnergySNN):
    named_params = {}
    for name, param in model.named_parameters():
        sm, lm, dm = (
            param.detach().clone(),
            0.0 * param.detach().clone(),
            0.0 * param.detach().clone(),
        )
        named_params[name] = (param, sm, lm, dm)
    return named_params


def post_optimizer_updates(named_params, alpha, beta):
    for name in named_params:
        param, sm, lm, _ = named_params[name]
        lm.data.add_(-alpha * (param - sm))
        sm.data.mul_((1.0 - beta))
        sm.data.add_(beta * param - (beta / alpha) * lm)


def get_regularizer_named_params(named_params, device, alpha, rho, _lambda=1.0):
    regularization = torch.zeros([], device=device)
    for name in named_params:
        param, sm, lm, _ = named_params[name]
        regularization += (rho - 1.0) * torch.sum(param * lm)
        r_p = _lambda * 0.5 * alpha * torch.sum(torch.square(param - sm))
        regularization += r_p
        # print(name,r_p)
    return regularization + 0.00001


def reset_named_params(named_params):
    for name in named_params:
        param, sm, lm, dm = named_params[name]
        param.data.copy_(sm.data)


def train_fptt_supervised(
    # models
    model: EnergySNN,
    # training
    T: int,  # number of timesteps
    K: int,  # number of updates
    update_interval: int,
    data_loader: DataLoader,
    model_params: dict,
    # optimization
    optimizer: torch.optim.Optimizer,
    alpha_energy: float,
    alpha_clf: float,
    clip_value: float,
    # fptt regularizer parameters
    alpha: float,
    beta: float,
    rho: float,
    # other
    file_name: str,
    timestamp: str,
    epoch: int,
    log_interval: int,
    debug=False,
):
    if debug:
        torch.autograd.set_detect_anomaly(True)

    train_loss = 0
    total_clf_loss = 0
    total_energy_loss = 0
    total_regularizaton_loss = 0

    # for each batch
    for i_batch, (data, labels) in enumerate(data_loader):
        # to device and reshape
        data, labels = data.to(model.device), labels.to(model.device)
        data = data.view(-1, model.d_in)
        B = labels.shape[0]
        h, readout = model.init_hidden(data.size(0))

        for t in range(T):
            if t % update_interval:
                h, readout = [value.detach() for value in h], readout.detach()

            h, readout = model.forward(data, h, readout)

            # only update model every omega steps
            if not (t % update_interval == 0 and t > 0):
                continue

            # loss calculations
            l_clf = (t + 1) / K * F.cross_entropy(readout, labels)
            l_energy = model.get_energies() / B
            l_reg = get_regularizer_named_params(model_params, model.device, alpha, rho, _lambda=1.0)
            loss = alpha_clf * l_clf + alpha_energy * l_energy + l_reg

            # decoder update
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            if clip_value > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            post_optimizer_updates(model_params, alpha, beta)

            train_loss += loss.item()
            total_clf_loss += l_clf.item()
            total_energy_loss += l_energy.item()
            total_regularizaton_loss += l_reg
            model.reset_energies()

        if (i_batch + 1) % log_interval == 0:
            print(
                (
                    "Train Epoch: {} batch {} | L_total: {:.2f} | L_E: {:.2f}"
                    + " | L_clf: {:.2f} | L_reg: {:.2f} | fr: {}"
                ).format(
                    epoch,
                    i_batch + 1,
                    train_loss / log_interval,
                    total_energy_loss / log_interval,
                    total_clf_loss / log_interval,
                    total_regularizaton_loss / log_interval,
                    [round(f / T / log_interval, 2) for f in model.firing_rates],
                )
            )

            train_loss = 0
            total_clf_loss = 0
            total_regularizaton_loss = 0
            total_energy_loss = 0
            model.reset_firing_rates()


def train_fptt_ssl(
    # models
    model: EnergySNN,
    # training
    T: int,  # number of timesteps
    K: int,  # number of updates
    update_interval: int,
    data_loader: DataLoader,
    model_params: dict,
    # optimization
    optimizer: torch.optim.Optimizer,
    alpha_energy: float,
    alpha_recon: float,
    clip_value: float,
    # fptt regularizer parameters
    alpha: float,
    beta: float,
    rho: float,
    # other
    file_name: str,
    timestamp: str,
    epoch: int,
    log_interval: int,
    debug=False,
):
    if debug:
        torch.autograd.set_detect_anomaly(True)

    train_loss = 0
    total_recon_loss = 0
    total_energy_loss = 0
    total_regularizaton_loss = 0

    # for each batch
    for i_batch, (data, labels) in enumerate(data_loader):
        # to device and reshape
        data, labels = data.to(model.device), labels.to(model.device)
        data = data.view(-1, model.d_in)
        B = labels.size()[0]
        h, readout = model.init_hidden(data.size(0))

        for t in range(T):
            if t % update_interval:
                h, readout = [value.detach() for value in h], readout.detach()

            h, readout = model.forward(data, h, readout)

            # only update model every omega steps
            if not (t % update_interval == 0 and t > 0):
                continue

            # loss calculations
            l_recon = (t + 1) / K * F.mse_loss(F.tanh(readout), data)
            l_energy = model.get_energies() / B
            l_reg = get_regularizer_named_params(model_params, model.device, alpha, rho, _lambda=1.0)
            loss = alpha_recon * l_recon + alpha_energy * l_energy + l_reg

            # decoder update
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            if clip_value > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            post_optimizer_updates(model_params, alpha, beta)

            train_loss += loss.item()
            total_recon_loss += l_recon.item()
            total_energy_loss += l_energy.item()
            total_regularizaton_loss += l_reg
            model.reset_energies()

        if (i_batch + 1) % log_interval == 0:
            fig, axs = plt.subplots(1, 10, figsize=(10, 1), dpi=200)
            for n in range(10):
                axs[n].imshow(readout[labels == n][0].detach().cpu().reshape(28, 28).numpy())
                axs[n].set_title(n)
                axs[n].axis("off")
            fig.savefig(
                f"images/reconstructions/{file_name}/{timestamp}/e={epoch}-b={i_batch}-loss={l_recon.item():.2f}.png"
            )
            plt.close()

            print(
                (
                    "Train Epoch: {} batch {} | L_total: {:.2f} | L_E: {:.2f}"
                    + " | L_rec: {:.2f} | L_reg: {:.2f} | fr: {}"
                ).format(
                    epoch,
                    i_batch + 1,
                    train_loss / log_interval,
                    total_energy_loss / log_interval,
                    total_recon_loss / log_interval,
                    total_regularizaton_loss / log_interval,
                    [round(f / T / log_interval, 2) for f in model.firing_rates],
                )
            )

            train_loss = 0
            total_recon_loss = 0
            total_regularizaton_loss = 0
            total_energy_loss = 0
            model.reset_firing_rates()
