import torch
import torch.nn.functional as F
from predcoding.snn.network import EnergySNN


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
    epoch,
    batch_size,
    log_interval,
    train_loader,
    model: EnergySNN,
    named_params,
    time_steps,
    k_updates,
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
):
    train_loss = 0
    total_clf_loss = 0
    total_regularizaton_loss = 0
    total_energy_loss = 0
    total_spike_loss = 0
    correct = 0
    model.train()

    # for each batch
    for batch_idx, (data, target) in enumerate(train_loader):

        # to device and reshape
        data, target = data.to(model.device), target.to(model.device)
        data = data.view(-1, model.d_in)

        B = target.size()[0]

        for t in range(time_steps):

            if t == 0:
                h, readout = model.init_hidden(data.size(0))
            else:
                h, readout = [v.detach() for v in h], readout.detach()

            o, h, readout = model.forward(data, h, readout)

            # get prediction
            if t == (time_steps - 1):
                pred = o.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            # only update model every omega steps
            if not (t % omega == 0 and t > 0):
                continue

            optimizer.zero_grad()

            # classification loss
            clf_loss = (t + 1) / k_updates * F.nll_loss(o, target)
            # clf_loss = snr*F.cross_entropy(output, target,reduction='none')
            # clf_loss = torch.mean(clf_loss)

            # regularizer loss
            regularizer = get_regularizer_named_params(named_params, model.device, alpha, rho, _lambda=1.0)

            # mem potential loss take l1 norm / num of neurons /batch size
            energy = model.get_energies() / B
            spike_loss = model.get_spike_loss(histories=h) / B

            # overall loss
            loss = clf_alpha * clf_loss + regularizer + energy_alpha * energy + spike_alpha * spike_loss

            loss.backward()

            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()
            post_optimizer_updates(named_params, alpha, beta)

            train_loss += loss.item()
            total_clf_loss += clf_loss.item()
            total_regularizaton_loss += regularizer  # .item()
            total_energy_loss += energy.item()
            total_spike_loss += spike_loss.item()

            model.reset_energies()

        if batch_idx > 0 and batch_idx % log_interval == (log_interval - 1):
            # print(
            #     "Train Epoch: {} [{}/{} ({:.0f}%)]\tenerg: {:.6f}\tlr: {:.6f}\ttrain acc:{:.4f}\tLoss: {:.6f}\
            #     \tClf: {:.6f}\tReg: {:.6f}\tFr_p: {:.6f}\tFr_r: {:.6f}".format(
            #         epoch,
            #         batch_idx * batch_size,
            #         len(train_loader.dataset),
            #         100.0 * batch_idx / len(train_loader),
            #         total_energy_loss / log_interval,
            #         lr,
            #         100 * correct / (log_interval * B),
            #         train_loss / log_interval,
            #         total_clf_loss / log_interval,
            #         total_regularizaton_loss / log_interval,
            #         model.firing_rates[0] / time_steps / log_interval,
            #         model.firing_rates[1] / time_steps / log_interval,
            #     )
            # )

            train_loss = 0
            total_clf_loss = 0
            total_regularizaton_loss = 0
            total_energy_loss = 0
            total_spike_loss = 0
            correct = 0
            model.reset_firing_rates()
