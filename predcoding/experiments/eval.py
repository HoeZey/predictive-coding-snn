import torch
import torch.nn.functional as F
from tqdm import tqdm
from predcoding.snn.network import EnergySNN


# test function
def test(model: EnergySNN, test_loader, time_steps):
    model.eval()
    test_loss = 0
    correct = 0

    # for data, target in test_loader:
    for data, target in test_loader:
        data, target = data.to(model.device), target.to(model.device)
        data = data.view(-1, model.d_in)

        with torch.no_grad():
            model.eval()
            h, readout = model.init_hidden(data.size(0))

            log_softmax_outputs, _ = model.inference(data, h, readout, time_steps)

            test_loss += F.nll_loss(log_softmax_outputs[-1], target, reduction="sum").data.item()

            pred = log_softmax_outputs[-1].data.max(1, keepdim=True)[1]

        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        torch.cuda.empty_cache()

    # wandb.log({'spike sequence': plot_spiking_sequence(hidden, target)})

    test_loss /= len(test_loader.dataset)
    test_acc = 100.0 * correct / len(test_loader.dataset)

    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({test_acc:.2f}%)")

    return test_loss, 100.0 * correct / len(test_loader.dataset)


def test_reconstruction(model: EnergySNN, test_loader, T):
    test_loss = 0.0
    for data, _ in test_loader:
        data = data.view(-1, model.d_in).to(model.device)
        h, readout = model.init_hidden(data.shape[0])
        with torch.no_grad():
            _, readout = model.inference(data, h, readout, T)
            test_loss += F.mse_loss(readout, data, reduction="sum").item()
        torch.cuda.empty_cache()
    return test_loss / len(test_loader.dataset)
