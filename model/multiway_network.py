import copy
import torch
import torch.nn as nn

class MultiwayNetwork(nn.Module):
    def __init__(self, module,num_channels, dim=1,concat_dim=1):
        super().__init__()
        self.dim = dim
        self.concat_dim = concat_dim
        self.multiway_modules = nn.ModuleList([
            copy.deepcopy(module) for _ in range(num_channels)
        ])
        for module in self.multiway_modules:
            module.reset_parameters()
        self.num_channels = num_channels
    def forward(self, x):
        x = torch.split(
            x,x.shape[self.dim]//self.num_channels,dim=self.dim,
        )
        y = [self.multiway_modules[i](x[i]) for i in range(self.num_channels)]
        return torch.cat(y, dim=self.concat_dim)