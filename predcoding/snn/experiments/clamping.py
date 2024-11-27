import torch
import torch.optim as optim
import numpy as np


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


def train_linear_proj(layer, model):
    mlp = MLP(hidden_dim[layer], 700, IN_dim).to(device)
    optimiser = optim.Adam(mlp.parameters(), lr=0.001, weight_decay=0.0001)

    loss_log = []

    for e in range(20):
        for i, (data, target) in enumerate(test_loader2):
            data, target = data.to(device), target.to(device)
            data = data.view(-1, model.in_dim)

            with torch.no_grad():
                model.eval()

                hidden = model.init_hidden(data.size(0))

                _, h = model.inference(data, hidden, T)

            spks = get_states(
                [h], 1 + layer * 4, hidden_dim[layer], batch_size, T, batch_size
            )

            train_data = torch.tensor(spks.mean(axis=1)).to(device)
            # print(train_data.size())

            optimiser.zero_grad()

            out = mlp(train_data)
            loss = MSE_loss(out, data)
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
