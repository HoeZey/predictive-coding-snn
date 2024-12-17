import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from predcoding.snn.network import EnergySNN
from predcoding.snn.hidden import LayerHidden


# linear decoder, but change the following class to other decoder types if necessary
class LinearDecoder(nn.Module):
    def __init__(self, d_in, d_out, device, d_hidden=None):
        super(LinearDecoder, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_hidden = d_hidden
        self.device = device

        self.fc1 = nn.Linear(d_in, d_out, device=device)
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


def get_states(h_hist: list[list[LayerHidden]], layer: int, d_hidden: int, batch_size, T=20, element="spike"):
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
    all_states = torch.zeros((batch_size, T, d_hidden))
    for t in range(T):
        if element == "spike":
            all_states[:, t] = h_hist[t][layer].spikes.detach()
        elif element == "dendrite":
            all_states[:, t] = h_hist[t][layer].dendrites.detach()
        elif element == "soma":
            all_states[:, t] = h_hist[t][layer].soma.detach()
    return all_states


def train_recon_decoder(
    epochs,
    layer,
    model: EnergySNN,
    data_loader,
    d_hidden,
    d_in,
    T,
    device,
    fn_loss=nn.MSELoss(),
):
    decoder = LinearDecoder(d_hidden, d_in, device)
    optimizer = optim.Adam(decoder.parameters(), lr=0.001, weight_decay=0.0001)
    model.eval()
    decoder.train()
    loss_log = []

    for _ in range(epochs):
        for _, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            data = data.view(-1, model.d_in)
            B = data.shape[0]

            with torch.no_grad():
                hidden, readout = model.init_hidden(data.size(0))
                h_hist, _ = model.inference(data, hidden, readout, T)
            spikes = get_states(h_hist, layer, d_hidden, B, T)
            reconstruction = decoder(spikes.mean(axis=1).to(device))

            loss = fn_loss(reconstruction, data)
            loss_log.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(f"L{layer + 1} final train loss: {loss:.4f}")

    return decoder, loss_log


def train_clf_decoder(
    model: EnergySNN,
    layer_to_decode,
    d_hidden,
    data_loader: DataLoader,
    epochs,
    T,
    device,
    fn_loss=nn.CrossEntropyLoss(),
):
    decoder = LinearDecoder(d_hidden, 10, device)
    optimizer = optim.Adam(decoder.parameters(), lr=0.001, weight_decay=0.001)
    model.eval()
    decoder.train()
    losses_all = []
    accs_all = []
    n_dataset = len(data_loader.dataset)
    print(f"Learning decoder for layer L{layer_to_decode + 1}/{len(model.d_hidden)}")
    for epoch in range(epochs):
        correct = 0
        losses = []
        for _, (data, labels) in enumerate(data_loader):
            data = data.view(-1, model.d_in).to(device)
            labels = labels.to(device)
            B = data.shape[0]

            with torch.no_grad():
                hidden, readout = model.init_hidden(B)
                h_hist, _ = model.inference(data, hidden, readout, T)

            spikes = get_states(h_hist, layer_to_decode, d_hidden, B, T)
            logits = decoder(spikes.mean(axis=1).to(device))

            loss = fn_loss(logits, labels)
            losses.append(loss.item())

            pred = logits.argmax(dim=-1)
            correct += (pred == labels).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = np.mean(losses)
        losses_all.append(epoch_loss)
        accs_all.append(100 * correct / n_dataset)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1} loss: {epoch_loss:.4f} | acc: {100 * correct / n_dataset: .2f}")

    return decoder, losses_all, accs_all
