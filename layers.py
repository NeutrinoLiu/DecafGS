import torch
from torch import nn
import time

class ExpLayer(nn.Module):
    """
    a nn module wapper for torch.exp
    """
    def __init__(self):
        super(ExpLayer, self).__init__()
    def forward(self, x):
        return torch.exp(x)

class IngoreCam(nn.Module):
    """
    ignore the last three dim of batched input features
    """
    def __init__(self):
        super(IngoreCam, self).__init__()
    def forward(self, x):
        return x[:, :-3]

def MLP_builder(in_dim, hidden_dim, out_dim, out_act, view_dependent=True):
    """
    build a 2 layer mlp module, refer to scaffold-gs
    """
    return nn.Sequential(
        nn.Identity() if view_dependent else IngoreCam(),
        nn.Linear(in_dim if view_dependent else in_dim - 3,
                  hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
        out_act
    )

class TempoMixture(nn.Module):
    def __init__(self,
                    in_dim,
                    hidden_dim,
                    out_dim,
                    further_dims,
                    skip,
                    depth,
                    cascade=True):       # insert skip connection at the input of skip-th layer
                                        # skip = 0 means no skip connection
        """
        build a 2 layer mlp module, refer to scaffold-gs
        """
        super(TempoMixture, self).__init__()
        assert skip < depth, "skip should be less than depth"
        self.skip = skip
        self.cascade = cascade
        self.layers = nn.ModuleList()
        for i in range(depth):
            sublayer_in_dim = in_dim if i == 0 else hidden_dim
            sublayer_out_dim = out_dim if i == depth - 1 else hidden_dim
            if i == skip and self.skip > 0:
                sublayer_in_dim += in_dim
            self.layers.append(
                nn.Sequential(
                    nn.Linear(sublayer_in_dim, sublayer_out_dim),
                    nn.ReLU()
                )
            )
        self.further_layers = nn.ModuleList(
            [nn.Linear(out_dim, further_dim) for further_dim in further_dims] if cascade else \
            [nn.Linear(in_dim, further_dim) for further_dim in further_dims]
        )
        for layer in self.further_layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
    def forward(self, x):
        original_x = x
        for i, layer in enumerate(self.layers):
            if i == self.skip and self.skip > 0:
                x = torch.cat([x, original_x], dim=-1)
            x = layer(x)

        further_x = []
        further_x_input = x if self.cascade else original_x
        for further_layer in self.further_layers:
            further_x.append(further_layer(further_x_input))
        return x, *further_x
