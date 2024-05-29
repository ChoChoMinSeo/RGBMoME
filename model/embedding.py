import torch
import torch.nn as nn
import numpy as np 

class PatchEmbedding(nn.Module):
    def __init__(
        self, 
        num_channels = 3,
        patch_size = 16,
        embed_dim = 384,
    ):
        super().__init__()
        patch_size = (patch_size, patch_size)
        
        self.proj = nn.Conv2d(
            num_channels, embed_dim, kernel_size = patch_size, stride = patch_size
        )
    def reset_parameters(self):
        self.proj.reset_parameters()
                
    def forward(self, x):      
        # split into patches  
        return self.proj(x).flatten(2).transpose(1,2)

class PatchEmbeddingByChannel(nn.Module):
    def __init__(
        self, 
        patch_size = 16,
        embed_dim = 384,
    ):
        super().__init__()
        patch_size = (patch_size, patch_size)
        
        self.proj = nn.Conv2d(
            1, embed_dim, kernel_size = patch_size, stride = patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
    def reset_parameters(self):
        self.proj.reset_parameters()
                
    def forward(self, x):      
        # split into patches
        x = self.proj(x).flatten(2).transpose(1,2)
        return torch.cat((self.cls_token.expand(x.size()[0],-1,-1),x),dim=1)
    
def get_sinusoid_encoding(num_tokens, token_len):
        """ Make Sinusoid Encoding Table

            Args:
                num_tokens (int): number of tokens
                token_len (int): length of a token
                
            Returns:
                (torch.FloatTensor) sinusoidal position encoding table
        """
        def get_position_angle_vec(i):
            return [i / np.power(10000, 2 * (j // 2) / token_len) for j in range(token_len)]

        sinusoid_table = np.array([get_position_angle_vec(i) for i in range(num_tokens)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) 

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class PositionalEmbedding1D_sep(nn.Module):
    # Different between patches and channel
    def __init__(self, seq_len, emb_dim, num_channels, relative = False):
        super().__init__()
        channel_length = emb_dim//num_channels
        self.num_channels = num_channels
        if relative:
            self.pos_embedding = nn.Parameter(get_sinusoid_encoding(num_tokens=seq_len, token_len=channel_length),requires_grad=False)
        else:
            self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, channel_length))
    def forward(self, x):
        # x: (batch_size, seq_len, emb_dim)        
        x+=self.pos_embedding.repeat(1,1,self.num_channels).type_as(x)
        return x
class PositionalEmbedding1D(nn.Module):
    # Different between patches and channel
    def __init__(self, seq_len, emb_dim, num_channels, relative = False):
        super().__init__()
        self.num_channels = num_channels
        if relative:
            self.pos_embedding = nn.Parameter(get_sinusoid_encoding(num_tokens=seq_len, token_len=emb_dim),requires_grad=False)
        else:
            self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, emb_dim))
    def forward(self, x):
        # x: (batch_size, seq_len, emb_dim)        
        x+=self.pos_embedding.type_as(x)
        return x

class ChannelEmbedding1D(nn.Module):
    # Same value for all patches. Different between channel
    def __init__(self, seq_len, emb_dim, num_channels):
        super().__init__()
        channel_length = emb_dim//num_channels
        self.channel_embedding = nn.Parameter(torch.cat([torch.zeros(1, 1, channel_length) for _ in range(num_channels)],dim=2))
        self.emb_dim = emb_dim
        self.seq_len = seq_len
    def forward(self, x):
        # x: (batch_size, seq_len, emb_dim)
        x += self.channel_embedding.expand(1,self.seq_len,self.emb_dim)
        return x