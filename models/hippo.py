import torch
from torch import multiply, nn
from cells.hippocell import HippoLegsCell
from models.mlp import Mlp
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Hippo(nn.Module):
    def __init__(self, N, gbt_alpha = 0.5, maxlength = 1024, reconst = False, output_size = 10):
        super(Hippo, self).__init__()
        self.hippo_cell_t = HippoLegsCell(N = N, gbt_alpha = gbt_alpha, maxlength = maxlength, reconst = reconst)
        self.mlp = Mlp(N, [output_size])
        self.N = N

    def forward(self, inputs, c_t = None):
        if c_t is None:
            c_t = torch.zeros(inputs.shape[1], 1, self.N).to(device)
        for t, f_t in enumerate(inputs):
            c_t, rec = self.hippo_cell_t(input = f_t, c_t = c_t, t = t)
            # shape c_t :-: (batchsize, 1, N)
            # squeeze singleton dim at index 1 for mlp input
        mlp_out = self.mlp(c_t.squeeze(1).float())
        return mlp_out, rec


