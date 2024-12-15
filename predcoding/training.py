import torch
import torch.nn.functional as F
from predcoding.snn.network import EnergySNN
from predcoding.experiments.decoder import LinearDecoder, get_states
from torch.utils.data import DataLoader


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
    return regularization


def reset_named_params(named_params):
    for name in named_params:
        param, sm, lm, dm = named_params[name]
        param.data.copy_(sm.data)


def train_fptt(
    epoch: int,
    batch_size: int,
    log_interval: int,
    train_loader: DataLoader,
    model: EnergySNN,
    named_params: dict,
    time_steps: int,
    k_updates: int,
    omega: int,
    optimizer: torch.optim.Optimizer,
    clf_alpha: float,
    energy_alpha: float,
    spike_alpha: float,
    clip: float,
    alpha: float,
    beta: float,
    rho: float,
    # decoder
    supervised=True,
    self_supervised=False,
    decoder: LinearDecoder = None,
    decoder_layer: int = None,
    recon_alpha: float = None,
    decoder_optimizer: torch.optim.Optimizer = None,
):
    assert supervised or self_supervised, "Training should have either supervised or self-supervised objective"

    torch.autograd.set_detect_anomaly(True)

    train_loss = 0
    total_clf_loss = 0
    total_recon_loss = 0
    total_regularizaton_loss = 0
    total_energy_loss = 0
    total_spike_loss = 0
    correct = 0

    # for each batch
    for i_batch, (data, labels) in enumerate(train_loader):
        # to device and reshape
        data, labels = data.to(model.device), labels.to(model.device)
        data = data.view(-1, model.d_in)
        B = labels.size()[0]
        h_hist = []
        h, readout = model.init_hidden(data.size(0))
        for t in range(time_steps):
            is_last_time_step = t == (time_steps - 1)
            if t % omega:
                h, readout = [value.detach() for value in h], readout.detach()

            log_preds, h, readout = model.forward(data, h, readout)
            h_hist.append(h)

            # get prediction
            if is_last_time_step:
                pred = log_preds.data.max(1, keepdim=True)[1]
                correct += pred.eq(labels.data.view_as(pred)).sum().item()

            # only update model every omega steps
            if not (t % omega == 0 and t > 0):
                continue

            # loss calculations
            l_clf = (t + 1) / k_updates * F.nll_loss(log_preds, labels)
            l_reg = get_regularizer_named_params(named_params, model.device, alpha, rho, _lambda=1.0)
            l_energy = model.get_energies() / B
            l_spike = model.get_spike_loss(histories=h) / B

            if is_last_time_step:
                spikes = get_states([h_hist], decoder_layer, model.d_hidden[decoder_layer], batch_size=B, T=t)
                reconstruction = decoder(spikes.mean(dim=1).to(decoder.device))
                l_recon = recon_alpha * F.mse_loss(reconstruction, data)

            loss = l_reg + energy_alpha * l_energy + spike_alpha * l_spike

            if supervised:
                loss = loss + clf_alpha * l_clf
            if self_supervised and is_last_time_step:
                loss = loss + l_recon

            # decoder update
            optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            if is_last_time_step:
                l_recon.backward(retain_graph=True)
                decoder_optimizer.step()

            # model update
            optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            post_optimizer_updates(named_params, alpha, beta)

            train_loss += loss.item()
            total_clf_loss += l_clf.item()
            if is_last_time_step:
                total_recon_loss += (l_recon).item() / recon_alpha
            total_regularizaton_loss += l_reg
            total_energy_loss += l_energy.item()
            total_spike_loss += l_spike.item()
            model.reset_energies()

        if (i_batch + 1) % log_interval == 0:
            print(
                (
                    "Train Epoch: {} [{}/{} ({:.0f}%)] train acc: {:.2f} | L_total: {:.2f} | L_E: {:.2f}"
                    + " | L_clf: {:.2f} | L_rec: {:.2f} | L_reg: {:.2f} | f_p: {:.2f} | f_r: {:.2f}"
                ).format(
                    epoch,
                    i_batch * batch_size,
                    len(train_loader.dataset),
                    100 * i_batch / len(train_loader),
                    100 * correct / (log_interval * B),
                    train_loss / log_interval,
                    total_energy_loss / log_interval,
                    total_clf_loss / log_interval,
                    total_recon_loss / log_interval,
                    total_regularizaton_loss / log_interval,
                    model.firing_rates[0] / time_steps / log_interval,
                    model.firing_rates[1] / time_steps / log_interval,
                )
            )

            train_loss = 0
            total_clf_loss = 0
            total_recon_loss = 0
            total_regularizaton_loss = 0
            total_energy_loss = 0
            total_spike_loss = 0
            correct = 0
            model.reset_firing_rates()


def train_fptt_bottleneck(
    # models
    model: EnergySNN,
    decoder: LinearDecoder,
    layer_to_decode: int,
    # training
    T: int,  # number of timesteps
    K: int,  # number of updates
    update_interval: int,
    data_loader: DataLoader,
    model_params: dict,
    decoder_params: dict,
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
        h_hist = torch.zeros((B, T, model.d_hidden[layer_to_decode]), device=model.device)
        h, readout = model.init_hidden(data.size(0))

        for t in range(T):
            # if t % update_interval:
            #     h, readout = [value.detach() for value in h], readout.detach()

            _, h, readout = model.forward(data, h, readout)

            # add spike to spike history
            h_hist[:, t] = h[layer_to_decode].spikes.detach()

            # only update model every omega steps
            if not (t % update_interval == 0 and t > 0):
                continue

            # loss calculations
            reconstruction = decoder(h_hist[:, : t + 1].mean(dim=1))
            l_recon = (t + 1) / K * F.mse_loss(reconstruction, data)
            l_energy = model.get_energies() / B
            l_reg = get_regularizer_named_params(model_params, model.device, alpha, rho, _lambda=1.0)
            loss = alpha_recon * l_recon + alpha_energy * l_energy + l_reg

            # decoder update
            optimizer.zero_grad()
            loss.backward()
            if clip_value > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            post_optimizer_updates(model_params, alpha, beta)
            post_optimizer_updates(decoder_params, alpha, beta)

            train_loss += loss.item()
            total_recon_loss += l_recon.item()
            total_energy_loss += l_energy.item()
            total_regularizaton_loss += l_reg
            model.reset_energies()

        if (i_batch + 1) % log_interval == 0:
            print(
                (
                    "Train Epoch: {} batch {} | L_total: {:.2f} | L_E: {:.2f}"
                    + " | L_clf: {:.2f} | L_rec: {:.2f} | L_reg: {:.2f} | fr: {:.2f}"
                ).format(
                    epoch,
                    i_batch + 1,
                    train_loss / log_interval,
                    total_energy_loss / log_interval,
                    total_clf_loss / log_interval,
                    total_recon_loss / log_interval,
                    total_regularizaton_loss / log_interval,
                    model.firing_rates / T / log_interval,
                )
            )

            train_loss = 0
            total_clf_loss = 0
            total_recon_loss = 0
            total_regularizaton_loss = 0
            total_energy_loss = 0
            model.reset_firing_rates()
