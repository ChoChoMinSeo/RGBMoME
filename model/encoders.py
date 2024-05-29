import torch.nn as nn
from .feedforward_network import FeedForwardNetwork
from .multihead_attention import MultiheadAttention
from .multiway_network import MultiwayNetwork

class EncoderLayer(nn.Module):
    def __init__(
        self,
        num_channels,
        embed_dim,
        num_heads,
        ffn_dim,
        activation_fn,
        ffn_dropout,
        dropout,
        is_per_channel_layer = True,
    ):
        super().__init__()
        dim_per_channel = embed_dim//num_channels
        if is_per_channel_layer:
            self.self_attn_layer_norm = MultiwayNetwork(
                nn.LayerNorm(dim_per_channel), num_channels, dim=2, concat_dim=2
            )
            self.final_layer_norm = MultiwayNetwork(
                nn.LayerNorm(dim_per_channel), num_channels, dim=2, concat_dim=2
            )
            self.ffn = MultiwayNetwork(
                FeedForwardNetwork(
                    dim_per_channel, ffn_dim, activation_fn, ffn_dropout
                ),num_channels,dim=2,concat_dim=2
            )
        else:
            self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
            
            self.final_layer_norm = nn.LayerNorm(embed_dim)
            self.ffn = FeedForwardNetwork(
                embed_dim, ffn_dim, activation_fn, ffn_dropout
            )
        self.self_attn = MultiheadAttention(
            embed_dim, num_heads
        )    
        self.dropout_module = nn.Dropout(dropout)          
        
    def forward(self, x):
        residual = x
        x = self.self_attn_layer_norm(x)
        
        x = self.self_attn(
            query = x,
            key = x,
            value = x,
        )
        
        x = self.dropout_module(x)
        
        x += residual
        residual = x
        
        x = self.final_layer_norm(x)
        
        x = self.ffn(x)
        x += residual
        
        return x
    
class Encoder(nn.Module):
    def __init__(
        self,
        sep_channel_depth,
        combine_channel_depth,
        num_channels,
        sep_embed_dim,
        comb_embed_dim,
        num_head,
        sep_ffn_dim,
        comb_ffn_dim,
        activation_fn,
        ffn_dropout = 0.1,
        dropout = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.sep_channel_depth = sep_channel_depth-1
        for _ in range(sep_channel_depth):
            self.layers.append(
                EncoderLayer(
                    num_channels, sep_embed_dim, num_head, sep_ffn_dim, activation_fn, ffn_dropout, 
                    dropout, is_per_channel_layer = True
                )
            )
        for _ in range(combine_channel_depth):
            self.layers.append(
                EncoderLayer(
                    num_channels, comb_embed_dim, num_head, comb_ffn_dim, activation_fn, ffn_dropout,
                    dropout, is_per_channel_layer = False
                )
            )
        
    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx==self.sep_channel_depth:
                aux_res = x
        return x,aux_res
        
        
        