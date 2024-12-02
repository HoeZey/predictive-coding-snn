import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


# linear decoder, but change the following class to other decoder types if necessary
class LinearReadout(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(LinearReadout, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(in_dim, out_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, out_dim)

        # # xavier initialisation
        nn.init.xavier_uniform_(self.fc1.weight)
        # nn.init.xavier_uniform_(self.fc2.weight)
        # nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        # x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        x = self.fc1(x)
        return x


def get_states(
    hiddens_all_: list, idx: int, hidden_dim_: int, batch_size, T=20, num_samples=10000
):
    """
    get a particular internal state depending on index passed to hidden

    Args:
        hidden_dim_: the size of a state, eg. num of r or p neurons
        T: total time steps
        hiddens_all_: list containing hidden states of all batch and time steps during inference
        idx: which index in h is taken out
    Returns:
        np.array containing desired states
    """

    all_states = []

    for batch_idx in range(len(hiddens_all_)):  # iterate over batch
        batch_ = []
        for t in range(T):
            seq_ = []
            for b in range(batch_size):
                seq_.append(hiddens_all_[batch_idx][t][idx][b].detach().cpu().numpy())
            seq_ = np.stack(seq_)
            batch_.append(seq_)
        batch_ = np.stack(batch_)

        all_states.append(batch_)

    all_states = np.stack(all_states)

    return all_states.transpose(0, 2, 1, 3).reshape(num_samples, T, hidden_dim_)


def train_linear_proj(
    layer,
    model,
    data_loader,
    d_hidden,
    d_in,
    batch_size,
    T,
    device,
    fn_loss=nn.MSELoss(),
):
    mlp = LinearReadout(d_hidden, 700, d_in).to(device)
    optimiser = optim.Adam(mlp.parameters(), lr=0.001, weight_decay=0.0001)

    loss_log = []

    for e in range(20):
        for i, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            data = data.view(-1, model.in_dim)

            with torch.no_grad():
                model.eval()

                hidden = model.init_hidden(data.size(0))

                _, h = model.inference(data, hidden, T)

            spks = get_states([h], 1 + layer * 4, d_hidden, batch_size, T, batch_size)

            train_data = torch.tensor(spks.mean(axis=1)).to(device)
            # print(train_data.size())

            optimiser.zero_grad()

            out = mlp(train_data)
            loss = fn_loss(out, data)
            loss_log.append(loss.data.cpu())

            loss.backward()
            optimiser.step()

        print("%i train loss: %.4f" % (e, loss))

        if e % 5 == 0:
            plt.imshow(out[target == 0][0].cpu().detach().reshape(28, 28))
            plt.title("sample1 %i" % target[target == 0][0].item())
            plt.show()

            # find the next image with class 0
            # plt.imshow(out[target == 0][1].cpu().detach().reshape(28, 28))
            # plt.title('sample2 %i' % target[target == 0][1].item())
            # plt.show()

    torch.cuda.empty_cache()

    mlp.eval()

    return mlp, [i.cpu() for i in loss_log]
