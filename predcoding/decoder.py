import torch
import torch.nn as nn
import matplotlib.pyplot as plt


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


def train_linear_proj(layer, model):
    mlp = LinearReadout(hidden_dim[layer], 700, IN_dim).to(device)
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
