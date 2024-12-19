import torch
import shutil


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(state, prefix, filename="_rec2_bias_checkpoint.pth.tar"):
    print("Saving at ", prefix + filename)
    torch.save(state, prefix + filename)


def model_result_dict_load(fn):
    """load tar file with saved model

    Args:
        fn (str): tar file name

    Returns:
        dict: dictornary containing saved results
    """
    with open(fn, "rb") as f:
        dict = torch.load(f)
    return dict
