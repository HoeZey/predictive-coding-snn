import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from predcoding.snn.layer import OutputLayer, SNNLayer
from predcoding.snn.hidden import LayerHidden


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

        self.firing_rates: list[float] = [0] * len(d_hidden)
        self.energies: list[torch.FloatTensor] = [0] * len(d_hidden)

        self.dropout = nn.Dropout(p_dropout)
        self.input_layer = nn.Linear(d_in, d_hidden[0], bias=bias, device=device)
        self.output_layer = OutputLayer(d_hidden[-1], d_out, is_fc=True)

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

        self.hidden_layers = nn.ModuleList(self.hidden_layers)

        self.forward_connections: list[nn.Linear] = [self.input_layer]
        self.backward_connections: list[nn.Linear] = []
        for d1, d2 in zip(d_hidden, d_hidden[1:]):
            self.forward_connections.append(nn.Linear(d1, d2, bias=bias, device=device))
            self.backward_connections.append(nn.Linear(d2, d1, bias=bias, device=device))
        self.backward_connections.append(nn.Linear(d_out, d_hidden[-1], bias=bias, device=device))

        self.forward_connections = nn.ModuleList(self.forward_connections)
        self.backward_connections = nn.ModuleList(self.backward_connections)

        for ff, fb in zip(self.forward_connections, self.backward_connections):
            nn.init.xavier_uniform_(ff.weight)
            nn.init.xavier_uniform_(fb.weight)
            if bias:
                nn.init.constant_(ff.bias, 0)
                nn.init.constant_(fb.bias, 0)

    def forward(
        self, x_t, histories: list[LayerHidden], readout: torch.FloatTensor
    ) -> tuple[torch.FloatTensor, list[LayerHidden], torch.FloatTensor]:
        batch_dim, input_size = x_t.shape
        spikes = self.dropout(x_t.reshape(batch_dim, input_size).float() * 0.5)

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

            h1 = LayerHidden(soma=soma, spikes=spikes, dendrites=dendrites, b=b)
            new_histories.append(h1)
            self.energies[i] = dendrites - soma
            self.firing_rates[i] += spikes.detach().mean().item()

        return new_histories, self.output_layer.forward(x_t=spikes, mem_t=readout)

    def inference(self, x, h, readout, T):
        """
        only called during inference
        :param x_t: input
        :param h: hidden states
        :param T: sequence length
        :param bystep: if true, then x_t is a sequence
        :return:
        """
        h_hist = []
        for _ in range(T):
            h, readout = self.forward(x, h, readout)
            h_hist.append(h)
        return readout, h_hist

    def init_hidden(self, n_batch, all_zero=False) -> tuple[list[LayerHidden], torch.FloatTensor]:
        histories = [
            LayerHidden.get_layer_history(n_batch, d, self.b0, self.device, all_zero=all_zero) for d in self.d_hidden
        ]
        return (histories, torch.zeros(n_batch, self.d_out, device=self.device))

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

            log_softmax, hidden_clamped, readout_clamped = self.forward(zeros, hidden_clamped, readout_clamped)

            log_softmax_hist.append(log_softmax)
            h_hist.append(hidden_clamped)

        return log_softmax_hist, h_hist

    def get_energies(self):
        l_energy = 0
        for e in self.energies:
            l_energy = l_energy + (e**2).sum()
        return l_energy / sum(self.d_hidden)

    def get_spike_loss(self, histories: list[LayerHidden]):
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
