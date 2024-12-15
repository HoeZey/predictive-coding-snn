import torch
from dataclasses import dataclass


@dataclass
class LayerHidden:
    soma: torch.FloatTensor | float
    spikes: torch.FloatTensor | float
    dendrites: torch.FloatTensor | float
    b: torch.FloatTensor | float

    @classmethod
    def get_layer_history(cls, n_batch: int, d: int, b0: float, device, all_zero=False):
        return cls(
            soma=(torch.rand(n_batch, d) if not all_zero else torch.zeros(n_batch, d)).to(device),
            spikes=torch.zeros(n_batch, d).to(device),
            dendrites=torch.zeros(n_batch, d).to(device),
            b=torch.full((n_batch, d), b0).to(device),
        )

    def detach(self):
        return LayerHidden(self.soma.detach(), self.spikes.detach(), self.dendrites.detach(), self.b.detach())
