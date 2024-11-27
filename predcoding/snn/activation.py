import torch
import math

b_j0 = 0.1  # neural threshold baseline
R_m = 3  # membrane resistance
gamma = 0.5  # gradient scale
lens = 0.5


def gaussian(x, mu=0.0, sigma=0.5):
    return (
        torch.exp(-((x - mu) ** 2) / (2 * sigma**2))
        / torch.sqrt(2 * torch.tensor(math.pi))
        / sigma
    )


class ActFun_adp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):  # input = membrane potential- threshold

        ctx.save_for_backward(input)

        return input.gt(0).float()  # is firing ???

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()

        # temp = abs(input) < lens

        scale = 6.0
        hight = 0.15

        # temp = torch.exp(-(input**2)/(2*lens**2))/torch.sqrt(2*torch.tensor(math.pi))/lens
        temp = (
            gaussian(input, mu=0.0, sigma=lens) * (1.0 + hight)
            - gaussian(input, mu=lens, sigma=scale * lens) * hight
            - gaussian(input, mu=-lens, sigma=scale * lens) * hight
        )

        # temp =  gaussian(input, mu=0., sigma=lens)

        return grad_input * temp.float() * gamma

        # return grad_input


act_fun_adp = ActFun_adp.apply
