import torch
import torch.nn.functional as F
from tqdm import tqdm
from predcoding.snn.network import EnergySNN


# test function
def test_clf(model: EnergySNN, test_loader, time_steps) -> tuple[float, float]:
    test_loss = 0
    correct = 0

    # for data, target in test_loader:
    for data, labels in test_loader:
        data, labels = data.to(model.device), labels.to(model.device)
        data = data.view(-1, model.d_in)

        h, logits = model.init_hidden(data.shape[0])
        with torch.no_grad():
            _, logits = model.inference(data, h, logits, time_steps)

        test_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(dim=-1) == labels).sum().item()

        torch.cuda.empty_cache()

    return test_loss / len(test_loader.dataset), correct / len(test_loader.dataset)


def test_reconstruction(model: EnergySNN, test_loader, T):
    test_loss = 0.0
    for data, _ in test_loader:
        data = data.view(-1, model.d_in).to(model.device)
        h, readout = model.init_hidden(data.shape[0])
        with torch.no_grad():
            _, readout = model.inference(data, h, readout, T)
            test_loss += F.mse_loss(F.tanh(readout), data, reduction="sum").item()
        torch.cuda.empty_cache()
    return test_loss / len(test_loader.dataset)
