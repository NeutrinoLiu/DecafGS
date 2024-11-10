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
        # nn.Linear(hidden_dim, hidden_dim),
        # nn.ReLU(),
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
                    decoupled=False):       # insert skip connection at the input of skip-th layer
                                        # skip = 0 means no skip connection
        """
        build a 2 layer mlp module, refer to scaffold-gs
        """
        super(TempoMixture, self).__init__()
        assert skip < depth, "skip should be less than depth"
        self.skip = skip
        self.decoupled = decoupled
        self.mlp_w_skip = SkippableMLP(in_dim, hidden_dim, out_dim, skip, depth)

        self.pre_delta_mlps = SkippableMLP(in_dim, hidden_dim, out_dim, skip, depth) if decoupled else None
        self.delta_mlps = nn.ModuleList(
            [ MLP_builder(
                in_dim=out_dim,
                hidden_dim=hidden_dim,
                out_dim=further_dim,
                out_act=nn.Identity()
                ) for further_dim in further_dims ]
        )
    def forward(self, x):
        feature = self.mlp_w_skip(x)
        delta_input = self.pre_delta_mlps(x) if self.decoupled else feature

        delta_outputs = []
        for mlp in self.delta_mlps:
            delta_outputs.append(mlp(delta_input))
        return feature, *delta_outputs

class SkippableMLP(nn.Module):
    def __init__(self,
                in_dim,
                hidden_dim,
                out_dim,
                skip,
                depth):       # insert skip connection at the input of skip-th layer
                                    # skip = 0 means no skip connection
        """
        build a 2 layer mlp module, refer to scaffold-gs
        """
        super(SkippableMLP, self).__init__()
        assert skip < depth, "skip should be less than depth"
        self.skip = skip
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
    def forward(self, x):
        original_x = x
        for i, layer in enumerate(self.layers):
            if i == self.skip and self.skip > 0:
                x = torch.cat([x, original_x], dim=-1)
            x = layer(x)
        return x