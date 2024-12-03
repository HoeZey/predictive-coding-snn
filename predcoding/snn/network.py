import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from predcoding.snn.layer import OutputLayer, SNNLayer


@dataclass
class LayerHistory:
    soma: torch.FloatTensor | float
    spikes: torch.FloatTensor | float
    dendrites: torch.FloatTensor | float
    b: torch.FloatTensor | float

    @classmethod
    def get_layer_history(cls, n_batch: int, d: int, b0: float, all_zero=False):
        return cls(
            soma=torch.rand(n_batch, d) if not all_zero else torch.zeros(n_batch, d),
            spikes=torch.zeros(n_batch, d),
            dendrites=torch.zeros(n_batch, d),
            b=torch.full((n_batch, d), b0),
        )

    def detach(self):
        return LayerHistory(self.soma.detach(), self.spikes.detach(), self.dendrites.detach(), self.b.detach())


class EnergySNN(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_hidden: list[int],
        d_out: int,
        is_adaptive: bool,
        one_to_one: bool,
        p_dropout: float,
        is_recurrent: bool,
        b0: float,
        device: str,
        bias=True,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.b0 = b0
        self.device = device

        self.firing_rates: list[torch.FloatTensor] = [0] * len(d_hidden)
        self.energies: list[torch.FloatTensor] = [0] * len(d_hidden)

        self.dropout = nn.Dropout(p_dropout)
        self.input_layer = nn.Linear(d_in, d_hidden[0], bias=bias)
        self.output_layer = OutputLayer(d_hidden[-1], d_out, is_fc=True, bias=bias)

        self.hidden_layers: list[SNNLayer] = []
        for d in d_hidden:
            self.hidden_layers.append(
                SNNLayer(
                    d_in=d,
                    d_hidden=d,
                    is_recurrent=is_recurrent,
                    is_adaptive=is_adaptive,
                    one_to_one=one_to_one,
                    device=device,
                    bias=bias,
                    b0=b0,
                )
            )

        self.forward_connections: list[nn.Linear] = [self.input_layer]
        self.backward_connections: list[nn.Linear] = []
        for d1, d2 in zip(d_hidden, d_hidden[1:]):
            self.forward_connections.append(nn.Linear(d1, d2, bias=bias))
            self.backward_connections.append(nn.Linear(d2, d1, bias=bias))
        self.backward_connections.append(nn.Linear(d_out, d_hidden[-1], bias=bias))

        for ff, fb in zip(self.forward_connections, self.backward_connections):
            nn.init.xavier_uniform_(ff.weight)
            nn.init.xavier_uniform_(fb.weight)
            if bias:
                nn.init.constant_(ff.bias, 0)
                nn.init.constant_(fb.bias, 0)

    def forward(
        self, x_t, histories: list[LayerHistory], readout: torch.FloatTensor
    ) -> tuple[torch.FloatTensor, list[LayerHistory], torch.FloatTensor]:
        batch_dim, input_size = x_t.shape
        x_t = self.dropout(x_t.reshape(batch_dim, input_size).float() * 0.5)
        spikes = x_t

        new_histories = []

        for i, (layer, ff_connections, fb_connections, h) in enumerate(
            zip(
                self.hidden_layers,
                self.forward_connections,
                self.backward_connections,
                histories,
            )
        ):
            is_last_layer = i + 1 == len(self.hidden_layers)
            fb_input = histories[i + 1].spikes if not is_last_layer else F.normalize(readout, dim=1)

            soma, spikes, dendrites, b = layer(
                ff=ff_connections(spikes),
                fb=fb_connections(fb_input),
                soma_t=h.soma,
                spike_t=h.spikes,
                a_curr_t=h.dendrites,
                b_t=h.b,
            )

            h1 = LayerHistory(soma=soma, spikes=spikes, dendrites=dendrites, b=b)
            new_histories.append(h1)
            self.energies[i] = dendrites - soma
            self.firing_rates[i] += spikes.detach().mean().item()

        readout = self.output_layer.forward(x_t=spikes, mem_t=readout)
        log_softmax = F.log_softmax(readout, dim=1)

        return log_softmax, new_histories, readout

    def inference(self, x_t, h, readout, T, bystep=None):
        """
        only called during inference
        :param x_t: input
        :param h: hidden states
        :param T: sequence length
        :param bystep: if true, then x_t is a sequence
        :return:
        """

        log_softmax_hist = []
        h_hist = []

        for t in range(T):
            if bystep is None:
                log_softmax, h, readout = self.forward(x_t, h, readout)
            else:
                log_softmax, h, readout = self.forward(x_t[t], h, readout)

            log_softmax_hist.append(log_softmax)
            h_hist.append((h, readout))

        return log_softmax_hist, h_hist

    def init_hidden(self, n_batch, all_zero=False) -> tuple[list[LayerHistory], torch.FloatTensor]:
        histories = [LayerHistory.get_layer_history(n_batch, d, self.b0, all_zero=all_zero) for d in self.d_hidden]
        return histories, torch.zeros(n_batch, self.d_out)
        # weight = next(self.parameters()).data
        # hidden_layer_weights = [
        #     [
        #         weight.new(n_batch, d).uniform_() if not all_zero else weight.new(n_batch, d).zero_(),
        #         weight.new(n_batch, d).zero_(),
        #         weight.new(n_batch, d).zero_(),
        #         weight.new(n_batch, d).fill_(self.b0),
        #     ]
        #     for d in self.d_hidden
        # ]
        # return [w for ws in hidden_layer_weights for w in ws] + [
        #     weight.new(n_batch, self.d_out).zero_(),  # layer out
        #     weight.new(n_batch, self.d_out).zero_(),  # sum spike
        # ]

    def clamped_generate(
        self,
        test_class,
        zeros,
        hidden_clamped,
        readout_clamped: torch.Tensor,
        T,
        clamp_value=0.5,
        noise=None,
    ):
        """
        generate representations with mem of read out clamped
        :param test_class: which class is clamped
        :param zeros: input containing zeros, absence of input
        :param h: hidden states
        :param T: sequence length
        :param noise: noise values
        :param index: index in h where noise is added to
        :return:
        """

        log_softmax_hist = []
        h_hist = []

        for _ in range(T):
            readout_clamped = readout_clamped.fill_(-clamp_value)
            readout_clamped[:, test_class] = clamp_value

            if noise is not None:
                readout_clamped[:] += noise

            log_softmax, hidden_clamped, _ = self.forward(zeros, hidden_clamped, readout_clamped)

            log_softmax_hist.append(log_softmax)
            h_hist.append(hidden_clamped)

        return log_softmax_hist, h_hist

    def get_energies(self):
        l_energy = 0
        for e in self.energies:
            l_energy = l_energy + (e**2).sum()
        return l_energy / sum(self.d_hidden)

    def get_spike_loss(self, histories: list[LayerHistory]):
        l_spikes = 0
        for h in histories:
            l_spikes = l_spikes + (h.spikes**2).sum()
        return l_spikes / sum(self.d_hidden)

    def reset_energies(self):
        for i in range(len(self.energies)):
            self.energies[i] = 0.0

    def reset_firing_rates(self):
        for i in range(len(self.firing_rates)):
            self.firing_rates[i] = 0.0


# 2 hidden layers
class SnnNetwork(nn.Module):
    def __init__(
        self,
        d_in: int,
        hidden_dims: list,
        out_dim: int,
        is_adapt: bool,
        one_to_one: bool,
        dp_rate: float,
        is_rec: bool,
        b0: float,
        device: str,
        bias=True,
    ):
        super(SnnNetwork, self).__init__()

        self.d_in = d_in
        self.d_hidden = hidden_dims
        self.d_out = out_dim
        self.is_adaptive = is_adapt
        self.one_to_one = one_to_one
        self.is_rec = is_rec
        self.b_j0 = b0
        self.device = device

        self.dp = nn.Dropout(dp_rate)

        self.layer1 = SNNLayer(
            hidden_dims[0],
            hidden_dims[0],
            is_recurrent=is_rec,
            is_adaptive=is_adapt,
            one_to_one=one_to_one,
            bias=bias,
            device=device,
            b0=b0,
        )

        # r in to r out
        self.layer1to2 = nn.Linear(hidden_dims[0], hidden_dims[1], bias=bias)
        nn.init.xavier_uniform_(self.layer1to2.weight)

        # r out to r in
        self.layer2to1 = nn.Linear(hidden_dims[1], hidden_dims[0], bias=bias)
        nn.init.xavier_uniform_(self.layer2to1.weight)

        self.layer2 = SNNLayer(
            hidden_dims[1],
            hidden_dims[1],
            is_recurrent=is_rec,
            is_adaptive=is_adapt,
            one_to_one=one_to_one,
            bias=bias,
            device=device,
            b0=b0,
        )

        self.output_layer = OutputLayer(hidden_dims[1], out_dim, is_fc=True, bias=bias)

        self.out2layer2 = nn.Linear(out_dim, hidden_dims[1], bias=bias)
        nn.init.xavier_uniform_(self.out2layer2.weight)

        if bias:
            nn.init.constant_(self.layer1to2.bias, 0)
            nn.init.constant_(self.layer2to1.bias, 0)
            nn.init.constant_(self.out2layer2.bias, 0)

        self.fr_layer2 = 0
        self.fr_layer1 = 0

        self.error1 = 0
        self.error2 = 0

    def forward(self, x_t, h: list[LayerHistory]):
        batch_dim, input_size = x_t.shape

        x_t = x_t.reshape(batch_dim, input_size).float()
        x_t = self.dp(x_t * 0.5)
        # poisson
        # x_t = x_t.gt(0.7).float()

        soma_1, spk_1, a_curr_1, b_1 = self.layer1(
            ff=x_t,
            fb=self.layer2to1(h[5]),
            soma_t=h[0],
            spk_t=h[1],
            a_curr_t=h[2],
            b_t=h[3],
        )

        self.error1 = a_curr_1 - soma_1

        # use out mem signal as feedback
        soma_2, spk_2, a_curr_2, b_2 = self.layer2(
            ff=self.layer1to2(spk_1),
            fb=self.out2layer2(F.normalize(h[-1], dim=1)),
            soma_t=h[4],
            spk_t=h[5],
            a_curr_t=h[6],
            b_t=h[7],
        )

        self.error2 = a_curr_2 - soma_2

        self.fr_layer2 = self.fr_layer2 + spk_2.detach().cpu().numpy().mean()
        self.fr_layer1 = self.fr_layer1 + spk_1.detach().cpu().numpy().mean()

        # read out from r_out neurons
        mem_out = self.output_layer(spk_2, h[-1])

        h = (soma_1, spk_1, a_curr_1, b_1, soma_2, spk_2, a_curr_2, b_2, mem_out)

        log_softmax = F.log_softmax(mem_out, dim=1)

        return log_softmax, h

    def inference(self, x_t, h, T, bystep=None):
        """
        only called during inference
        :param x_t: input
        :param h: hidden states
        :param T: sequence length
        :param bystep: if true, then x_t is a sequence
        :return:
        """

        log_softmax_hist = []
        h_hist = []

        for t in range(T):
            if bystep is None:
                log_softmax, h = self.forward(x_t, h)
            else:
                log_softmax, h = self.forward(x_t[t], h)

            log_softmax_hist.append(log_softmax)
            h_hist.append(h)

        return log_softmax_hist, h_hist

    def clamped_generate(
        self,
        test_class,
        zeros,
        h_clamped,
        T,
        clamp_value=0.5,
        batch=False,
        noise=None,
    ):
        """
        generate representations with mem of read out clamped
        :param test_class: which class is clamped
        :param zeros: input containing zeros, absence of input
        :param h: hidden states
        :param T: sequence length
        :return:
        """

        log_softmax_hist = []
        h_hist = []

        for t in range(T):
            if not batch:
                h_clamped[-1][0] = -clamp_value
                h_clamped[-1][0, test_class] = clamp_value
            else:
                h_clamped[-1][:, :] = torch.full(h_clamped[-1].size(), -clamp_value).to(self.device)
                h_clamped[-1][:, test_class] = clamp_value

            if noise is not None:
                h_clamped[-1][:] += noise

            # if t==0:
            #     print(h_clamped[-1])

            log_softmax, h_clamped = self.forward(zeros, h_clamped)

            log_softmax_hist.append(log_softmax)
            h_hist.append(h_clamped)

        return log_softmax_hist, h_hist

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (
            # r
            weight.new(bsz, self.d_hidden[0]).uniform_(),
            weight.new(bsz, self.d_hidden[0]).zero_(),
            weight.new(bsz, self.d_hidden[0]).zero_(),
            weight.new(bsz, self.d_hidden[0]).fill_(self.b_j0),
            # p
            weight.new(bsz, self.d_hidden[1]).uniform_(),
            weight.new(bsz, self.d_hidden[1]).zero_(),
            weight.new(bsz, self.d_hidden[1]).zero_(),
            weight.new(bsz, self.d_hidden[1]).fill_(self.b_j0),
            # layer out
            weight.new(bsz, self.d_out).zero_(),
            # sum spike
            weight.new(bsz, self.d_out).zero_(),
        )

    def get_energy(self):
        return torch.sum(self.error1**2) + torch.sum(self.error2**2)

    def get_spike_loss(self, h):
        return torch.sum(h[1]) + torch.sum(h[5])

    def reset_errors(self):
        self.error1 = 0
        self.error2 = 0


# 3 hidden layers
class SnnNetwork3Layer(SnnNetwork):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: list,
        out_dim: int,
        is_adapt: bool,
        one_to_one: bool,
        dp_rate: float,
        is_rec: bool,
        b0: float,
        device: str,
        bias=True,
    ):
        super().__init__(
            in_dim,
            hidden_dims,
            out_dim,
            is_adapt,
            one_to_one,
            dp_rate,
            is_rec,
            b0,
            device,
        )

        self.layer3 = SNNLayer(
            hidden_dims[2],
            hidden_dims[2],
            is_recurrent=is_rec,
            is_adaptive=is_adapt,
            one_to_one=one_to_one,
            device=device,
            bias=bias,
            b0=b0,
        )

        self.layer2to3 = nn.Linear(hidden_dims[1], hidden_dims[2], bias=bias)
        nn.init.xavier_uniform_(self.layer2to3.weight)

        # r out to r in
        self.layer3to2 = nn.Linear(hidden_dims[2], hidden_dims[1], bias=bias)
        nn.init.xavier_uniform_(self.layer3to2.weight)

        self.output_layer = OutputLayer(hidden_dims[2], out_dim, is_fc=True)

        self.out2layer3 = nn.Linear(out_dim, hidden_dims[2], bias=bias)
        nn.init.xavier_uniform_(self.out2layer3.weight)

        self.fr_layer3 = 0

        self.error3 = 0

        self.input_fc = nn.Linear(in_dim, hidden_dims[0], bias=bias)
        nn.init.xavier_uniform_(self.input_fc.weight)

        if bias:
            nn.init.constant_(self.layer2to3.bias, 0)
            nn.init.constant_(self.layer3to2.bias, 0)
            nn.init.constant_(self.out2layer3.bias, 0)
            nn.init.constant_(self.input_fc.bias, 0)
            print("bias set to 0")

    def forward(self, x_t, h):
        batch_dim, input_size = x_t.shape

        x_t = x_t.reshape(batch_dim, input_size).float()
        x_t = self.dp(x_t)
        # poisson
        # x_t = x_t.gt(0.7).float()
        x_t = self.input_fc(x_t)

        soma_1, spk_1, a_curr_1, b_1 = self.layer1(
            ff=x_t,
            fb=self.layer2to1(h[5]),
            soma_t=h[0],
            spk_t=h[1],
            a_curr_t=h[2],
            b_t=h[3],
        )

        self.error1 = a_curr_1 - soma_1

        # use out mem signal as feedback
        soma_2, spk_2, a_curr_2, b_2 = self.layer2(
            ff=self.layer1to2(spk_1),
            fb=self.layer3to2(h[9]),
            soma_t=h[4],
            spk_t=h[5],
            a_curr_t=h[6],
            b_t=h[7],
        )

        self.error2 = a_curr_2 - soma_2

        soma_3, spk_3, a_curr_3, b_3 = self.layer3(
            ff=self.layer2to3(spk_2),
            fb=self.out2layer3(F.normalize(h[-1], dim=1)),
            soma_t=h[8],
            spk_t=h[9],
            a_curr_t=h[10],
            b_t=h[11],
        )

        self.error3 = a_curr_3 - soma_3

        self.fr_layer3 += spk_3.detach().cpu().numpy().mean()
        self.fr_layer2 += spk_2.detach().cpu().numpy().mean()
        self.fr_layer1 += spk_1.detach().cpu().numpy().mean()

        # read out from r_out neurons
        mem_out = self.output_layer(spk_3, h[-1])

        h = (
            {
                "soma": soma_1,
                "spikes": spk_1,
                "current": a_curr_1,
                "bias": b_1,
            },
            {
                "soma": soma_2,
                "spikes": spk_2,
                "current": a_curr_2,
                "bias": b_2,
            },
            {
                "soma": soma_3,
                "spikes": spk_3,
                "current": a_curr_3,
                "bias": b_3,
            },
        )

        log_softmax = F.log_softmax(mem_out, dim=1)

        return log_softmax, h, mem_out

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (
            # l1
            weight.new(bsz, self.d_hidden[0]).uniform_(),
            weight.new(bsz, self.d_hidden[0]).zero_(),
            weight.new(bsz, self.d_hidden[0]).zero_(),
            weight.new(bsz, self.d_hidden[0]).fill_(self.b_j0),
            # l2
            weight.new(bsz, self.d_hidden[1]).uniform_(),
            weight.new(bsz, self.d_hidden[1]).zero_(),
            weight.new(bsz, self.d_hidden[1]).zero_(),
            weight.new(bsz, self.d_hidden[1]).fill_(self.b_j0),
            # l3
            weight.new(bsz, self.d_hidden[2]).uniform_(),
            weight.new(bsz, self.d_hidden[2]).zero_(),
            weight.new(bsz, self.d_hidden[2]).zero_(),
            weight.new(bsz, self.d_hidden[2]).fill_(self.b_j0),
            # layer out
            weight.new(bsz, self.d_out).zero_(),
            # sum spike
            weight.new(bsz, self.d_out).zero_(),
        )

    def init_hidden_allzero(self, bsz):
        weight = next(self.parameters()).data
        return (
            # l1
            weight.new(bsz, self.d_hidden[0]).zero_(),
            weight.new(bsz, self.d_hidden[0]).zero_(),
            weight.new(bsz, self.d_hidden[0]).zero_(),
            weight.new(bsz, self.d_hidden[0]).fill_(self.b_j0),
            # l2
            weight.new(bsz, self.d_hidden[1]).zero_(),
            weight.new(bsz, self.d_hidden[1]).zero_(),
            weight.new(bsz, self.d_hidden[1]).zero_(),
            weight.new(bsz, self.d_hidden[1]).fill_(self.b_j0),
            # l3
            weight.new(bsz, self.d_hidden[2]).zero_(),
            weight.new(bsz, self.d_hidden[2]).zero_(),
            weight.new(bsz, self.d_hidden[2]).zero_(),
            weight.new(bsz, self.d_hidden[2]).fill_(self.b_j0),
            # layer out
            weight.new(bsz, self.d_out).zero_(),
            # sum spike
            weight.new(bsz, self.d_out).zero_(),
        )

    def clamp_generate(
        self,
        test_class,
        zeros,
        hidden_clamped,
        T,
        device,
        clamp_value=0.5,
        batch=False,
        index=None,
        noise=None,
    ):
        """
        generate representations with mem of read out clamped
        :param test_class: which class is clamped
        :param zeros: input containing zeros, absence of input
        :param h: hidden states
        :param T: sequence length
        :param noise: noise values
        :param index: index in h where noise is added to
        :return:
        """

        log_softmax_hist = []
        h_hist = []

        for _ in range(T):
            if not batch:
                hidden_clamped[-1][0] = -clamp_value
                hidden_clamped[-1][0, test_class] = clamp_value
            else:
                hidden_clamped[-1][:, :] = torch.full(hidden_clamped[-1].size(), -clamp_value).to(device)
                hidden_clamped[-1][:, test_class] = clamp_value

            if noise is not None:
                if index is not None:
                    hidden_clamped[index][:, :] += noise * hidden_clamped[index][:, :]
                else:
                    hidden_clamped[-1][:] += noise

            log_softmax, hidden_clamped, _ = self.forward(zeros, hidden_clamped)

            log_softmax_hist.append(log_softmax)
            h_hist.append(hidden_clamped)

        return log_softmax_hist, h_hist

    def get_energy(self):
        return (
            torch.sum(torch.abs(self.error1)) + torch.sum(torch.abs(self.error2)) + torch.sum(torch.abs(self.error3))
        ) / sum(self.d_hidden)

    def get_spike_loss(self, h):
        return (torch.sum(h[1]) + torch.sum(h[5]) + torch.sum(h[9])) / sum(self.d_hidden)

    def reset_errors(self):
        self.error1 = 0
        self.error2 = 0
        self.error3 = 0
