import torch
from torch import nn
import time
from resfield import Linear

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

def MLP_builder(in_dim, hidden_dim, out_dim, out_act, view_dependent=True, T_max=None, deeper=False):
    """
    build a 2 layer mlp module, refer to scaffold-gs
    """
    if T_max is not None:
        if deeper:
            return ResSequential(
                nn.Identity() if view_dependent else IngoreCam(),
                ResLinear(in_dim if view_dependent else in_dim - 3, hidden_dim, T_max),
                nn.ReLU(),
                ResLinear(hidden_dim, hidden_dim, T_max),
                nn.ReLU(),
                ResLinear(hidden_dim, out_dim, T_max),
                out_act
            )
        else:
            return ResSequential(
                nn.Identity() if view_dependent else IngoreCam(),
                ResLinear(in_dim if view_dependent else in_dim - 3, hidden_dim, T_max),
                nn.ReLU(),
                ResLinear(hidden_dim, out_dim, T_max),
                out_act
            )
    else:
        return nn.Sequential(
            nn.Identity() if view_dependent else IngoreCam(),
            nn.Linear(in_dim if view_dependent else in_dim - 3, hidden_dim),
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
                    decoupled=False,
                    T_max=None):       # insert skip connection at the input of skip-th layer
                                        # skip = 0 means no skip connection
        """
        build a 2 layer mlp module, refer to scaffold-gs
        """
        super(TempoMixture, self).__init__()
        assert skip < depth, "skip should be less than depth"
        self.resfield = True if T_max is not None else False
        self.skip = skip
        self.decoupled = decoupled

        self.embed_mixing_mlp = SkippableMLP(in_dim, hidden_dim, out_dim, skip, depth, T_max)
        self.embed_mixing_mlp_2 = SkippableMLP(in_dim, hidden_dim, out_dim, skip, depth, T_max) if decoupled else None
        self.delta_mlps = nn.ModuleList(
            [ MLP_builder(
                in_dim=out_dim,
                hidden_dim=hidden_dim,
                out_dim=further_dim,
                out_act=nn.Identity(),
                T_max=T_max
                ) for further_dim in further_dims ]
        )
    def forward(self, x, t=None):
        assert self.resfield == (t is not None), "t should be provided if and only if resfield is True"
        feature = self.embed_mixing_mlp(x, t)
        delta_input = self.embed_mixing_mlp_2(x, t) if self.decoupled else feature

        delta_outputs = []
        for mlp in self.delta_mlps:
            delta_outputs.append(
                mlp(delta_input, t) if isinstance(mlp, ResMLP) else mlp(delta_input)
                )
        return feature, *delta_outputs

class SkippableMLP(nn.Module):
    def __init__(self,
                in_dim,
                hidden_dim,
                out_dim,
                skip,
                depth,
                T_max=None):       # insert skip connection at the input of skip-th layer
                                    # skip = 0 means no skip connection
        """
        build a 2 layer mlp module, refer to scaffold-gs
        """
        super(SkippableMLP, self).__init__()
        assert skip < depth, "skip should be less than depth"
        self.skip = skip
        self.layers = nn.ModuleList()
        self.resfield = True if T_max is not None else False
        for i in range(depth):
            sublayer_in_dim = in_dim if i == 0 else hidden_dim
            sublayer_out_dim = out_dim if i == depth - 1 else hidden_dim
            if i == skip and self.skip > 0:
                sublayer_in_dim += in_dim
            if T_max is not None:
                new_layer = ResSequential(
                    ResLinear(sublayer_in_dim, sublayer_out_dim, T_max),
                    nn.ReLU())
            else:
                new_layer = nn.Sequential(
                    nn.Linear(sublayer_in_dim, sublayer_out_dim),
                    nn.ReLU())
            self.layers.append(new_layer)
    def forward(self, x, t=None):
        assert self.resfield == (t is not None), "t should be provided if and only if resfield is True"
        original_x = x
        for i, layer in enumerate(self.layers):
            if i == self.skip and self.skip > 0:
                x = torch.cat([x, original_x], dim=-1)
            x = layer(x, t) if isinstance(layer, ResMLP) else layer(x)
        return x


class ResMLP(nn.Module):
    def __init__(self):
        super(ResMLP, self).__init__()
        pass
    def forward(self, x):
        raise NotImplementedError("abstract method")

# class ResLinear(ResMLP):
#     def __init__(self, in_dim, out_dim, T_max, rank=16):
#         super(ResLinear, self).__init__()
#         self.weight_base = nn.Parameter(torch.randn(in_dim, out_dim)* 0.01 ) 
#         self.weight_bank = nn.Parameter(torch.randn(rank, in_dim, out_dim)* 0.01 ) 
#         self.weight_coeff = nn.Parameter(torch.randn(T_max, rank)* 0.01 ) 
#         self.bias = nn.Parameter(torch.randn(out_dim)* 0.01 ) 
#     def forward(self, x, t):
#         assert t < self.weight_coeff.shape[0], "t should be less than T_max"
#         weight = torch.einsum('w, wnd -> nd', self.weight_coeff[t], self.weight_bank)
#         return torch.matmul(x, weight) + self.bias

class ResLinear(ResMLP):
    def __init__(self, in_dim, out_dim, T_max, rank=16):
        super(ResLinear, self).__init__()
        self.layer = Linear(in_dim, out_dim, rank=rank, capacity=T_max)
        self.span = T_max-1
    def forward(self, x, t):
        assert t <= self.span, "t should be less than T_max"
        unified_time = 2 * t / self.span - 1 
        return self.layer(x, input_time=unified_time, frame_id=t)

class ResSequential(ResMLP):
    def __init__(self, *layers):
        super(ResSequential, self).__init__()
        self.layers = nn.ModuleList(layers)
    def forward(self, x, t):
        for layer in self.layers:
            if isinstance(layer, ResMLP):
                x = layer(x, t)
            else:
                x = layer(x)
        return x