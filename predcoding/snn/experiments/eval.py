import torch
from tqdm import tqdm


# test function
def test(model, test_loader, time_steps, device):
    model.eval()
    test_loss = 0
    correct = 0

    # for data, target in test_loader:
    for data, target in tqdm(test_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(-1, model.in_dim)

        with torch.no_grad():
            model.eval()
            hidden = model.init_hidden(data.size(0))

            log_softmax_outputs, hidden = model.inference(data, hidden, time_steps)

            test_loss += F.nll_loss(
                log_softmax_outputs[-1], target, reduction="sum"
            ).data.item()

            pred = log_softmax_outputs[-1].data.max(1, keepdim=True)[1]

        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        torch.cuda.empty_cache()

    # wandb.log({'spike sequence': plot_spiking_sequence(hidden, target)})

    test_loss /= len(test_loader.dataset)
    test_acc = 100.0 * correct / len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), test_acc
        )
    )

    return test_loss, 100.0 * correct / len(test_loader.dataset)
