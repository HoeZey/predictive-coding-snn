import torch
import torch.nn as nn
from predcoding.snn.activation import act_fun_adp


# layers
def shifted_sigmoid(currents):
    return (1 / (1 + torch.exp(-currents)) - 0.5) / 2


class SNNLayer(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        is_recurrent: bool,
        is_adaptive: bool,
        one_to_one: bool,
        b0,
        device,
        tau_m_init=15.0,
        tau_adap_init=20,
        tau_a_init=15.0,
        dt=0.5,
        bias=True,
    ):
        super(SNNLayer, self).__init__()

        self.is_recurrent = is_recurrent
        self.is_adaptive = is_adaptive
        self.one_to_one = one_to_one
        self.dt = dt
        self.baseline_threshold = b0
        self.device = device

        if is_recurrent:
            self.rec_w = nn.Linear(d_hidden, d_hidden, bias=bias, device=device)
            # init weights
            if bias:
                nn.init.constant_(self.rec_w.bias, 0)
            nn.init.xavier_uniform_(self.rec_w.weight)

            p = torch.full(self.rec_w.weight.size(), fill_value=0.5).to(device)
            self.weight_mask = torch.bernoulli(p)

        else:
            self.fc_weights = nn.Linear(d_in, d_hidden, bias=bias, device=device)
            if bias:
                nn.init.constant_(self.fc_weights.bias, 0)
            nn.init.xavier_uniform_(self.fc_weights.weight)

        # define param for time constants
        self.tau_adp = nn.Parameter(torch.Tensor(d_hidden))
        self.tau_m = nn.Parameter(torch.Tensor(d_hidden))
        self.tau_a = nn.Parameter(torch.Tensor(d_hidden))

        nn.init.normal_(self.tau_adp, tau_adap_init, 0.1)
        nn.init.normal_(self.tau_m, tau_m_init, 0.1)
        nn.init.normal_(self.tau_a, tau_a_init, 0.1)

        self.sigmoid = nn.Sigmoid()

    def mem_update(self, ff, fb, soma, spike, a_curr, b, is_adapt, r_m=3):
        """
        mem update for each layer of neurons
        :param ff: feedforward signal
        :param fb: feedback signal to apical tuft
        :param soma: mem voltage potential at soma
        :param spike: spiking at last time step
        :param a_curr: apical tuft current at last t
        :param b: adaptive threshold
        :return:
        """
        # alpha = self.sigmoid(self.tau_m)
        # rho = self.sigmoid(self.tau_adp)
        # eta = self.sigmoid(self.tau_a)
        alpha = torch.exp(-self.dt / self.tau_m).to(self.device)
        rho = torch.exp(-self.dt / self.tau_adp).to(self.device)
        eta = torch.exp(-self.dt / self.tau_a).to(self.device)

        if is_adapt:
            beta = 1.8
        else:
            beta = 0.0

        b = rho * b + (1 - rho) * spike  # adaptive contribution
        new_thre = self.baseline_threshold + beta * b  # udpated threshold

        a_new = eta * a_curr + fb  # fb into apical tuft

        soma_new = alpha * soma + shifted_sigmoid(a_new) + ff - new_thre * spike
        # soma_new = alpha * soma + 1/2 * (a_new) + ffs - new_thre * spike

        inputs_ = soma_new - new_thre

        spike = act_fun_adp(inputs_)  # act_fun : approximation firing function
        # mem = (1 - spike) * mem

        return soma_new, spike, a_new, new_thre, b

    def forward(self, ff, fb, soma_t, spike_t, a_curr_t, b_t):
        """
        forward function of a single layer. given previous neuron states and current input, update neuron states

        :param ff: ff signal (not counting rec)
        :param fb: fb top down signal
        :param soma_t: soma voltage
        :param a_curr_t: apical tuft voltage
        :return:
        """

        if self.is_recurrent:
            self.rec_w.weight.data = self.rec_w.weight.data * self.weight_mask
            # self.rec_w.weight.data = (self.rec_w.weight.data < 0).float() * self.rec_w.weight.data
            r_in = ff + self.rec_w(spike_t)
        else:
            if self.one_to_one:
                r_in = ff
            else:
                r_in = self.fc_weights(ff)

        soma_t1, spike_t1, a_curr_t1, _, b_t1 = self.mem_update(
            r_in, fb, soma_t, spike_t, a_curr_t, b_t, self.is_adaptive
        )

        return soma_t1, spike_t1, a_curr_t1, b_t1


class OutputLayer(nn.Module):
    def __init__(self, d_in: int, d_out: int, is_fc: bool, tau_fixed=None, bias=True, dt=0.5):
        """
        output layer class
        :param is_fc: whether integrator is fc to r_out in rec or not
        """
        super(OutputLayer, self).__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.is_fc = is_fc
        self.dt = dt

        if is_fc:
            self.fc = nn.Linear(d_in, d_out, bias=bias)
            nn.init.xavier_uniform_(self.fc.weight)
            if bias:
                nn.init.constant_(self.fc.bias, 0)

        # tau_m
        if tau_fixed is None:
            self.tau_m = nn.Parameter(torch.Tensor(d_out))
            nn.init.constant_(self.tau_m, 5)
        else:
            self.tau_m = nn.Parameter(torch.Tensor(d_out), requires_grad=False)
            nn.init.constant_(self.tau_m, tau_fixed)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_t, mem_t):
        """
        integrator neuron without spikes
        """
        alpha = torch.exp(-self.dt / self.tau_m)
        # alpha = self.sigmoid(self.tau_m)

        if self.is_fc:
            x_t = self.fc(x_t)
        else:
            x_t = x_t.view(-1, 10, int(self.d_in / 10)).mean(dim=2)  # sum up population spike

        # d_mem = -soma_t + x_t
        mem = (mem_t + x_t) * alpha
        # mem = alpha * soma_t + (1 - alpha) * x_t
        return mem
