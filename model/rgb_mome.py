import torch.nn as nn
from .embedding import PatchEmbeddingByChannel, PositionalEmbedding1D,PositionalEmbedding1D_sep,ChannelEmbedding1D,PatchEmbedding
from .multiway_network import MultiwayNetwork
from .encoders import Encoder


class RGB_MoME(nn.Module):
    # Channelwise layer + Original ViT Layer
    def __init__(
        self,
        num_channels = 3,
        img_size = 224,
        patch_size = 16,
        emb_dim = 768,
        num_head = 12,
        sep_ffn_dim = 1024,
        comb_ffn_dim = 3072,
        sep_channel_depth = 3,
        combine_channel_depth = 9,
        num_classes = 10,
        ffn_dropout = 0.1, # after ffn -> dropout
        dropout = 0.1, # after MSA -> dropout
        activation_fn = 'gelu',
        task_type = 'classification',
        include_channel_embedding = True,
        relative_pos = False,
        multiway_patch_embedding = True,
        auxilary_output = False,
    ):
        super().__init__()
        seq_len = (img_size//patch_size)**2+1
        self.num_channels = num_channels
        self.task_type = task_type
        self.contain_channel_embedding = include_channel_embedding
        self.auxilary_output = auxilary_output
        if multiway_patch_embedding:
            self.patch_embedding = MultiwayNetwork(
                PatchEmbeddingByChannel(
                    patch_size = patch_size,
                    embed_dim = emb_dim//num_channels,
                ),num_channels, dim = 1, concat_dim = 2
            )
            self.positional_embedding = PositionalEmbedding1D_sep(seq_len=seq_len,emb_dim=emb_dim, num_channels=num_channels,relative=relative_pos)
        else:
            self.patch_embedding = PatchEmbedding(num_channels, patch_size, emb_dim)
            self.positional_embedding = PositionalEmbedding1D(seq_len=seq_len,emb_dim=emb_dim, num_channels=num_channels,relative=relative_pos)
        
        if include_channel_embedding:
            self.channel_embedding = ChannelEmbedding1D(seq_len=seq_len,emb_dim=emb_dim,num_channels=num_channels)
        
        self.encoder = Encoder(
            sep_channel_depth, combine_channel_depth, num_channels, emb_dim, emb_dim, num_head, sep_ffn_dim, comb_ffn_dim, 
            activation_fn, ffn_dropout, dropout
        )
        if task_type == 'classification':
            self.task_head = nn.Linear(emb_dim, num_classes)
            if auxilary_output:
                self.aux_task_head = MultiwayNetwork(
                    nn.Linear(emb_dim//num_channels,num_classes),num_channels,dim=1,concat_dim=1
                )
        self.identity = nn.Identity()
    def forward(
        self, x = None, # input image B,C,H,W
    ):        
        x = self.patch_embedding(x)
        
        x = self.positional_embedding(x)
        
        if self.contain_channel_embedding:
            x = self.channel_embedding(x)
        
        x,aux_res = self.encoder(x)
        
        if self.task_type == 'classification':
            x = x[:,0]
            x = self.identity(x)
            x = self.task_head(x)
            if self.auxilary_output:
                aux_res = aux_res[:,0]
                aux_res = self.identity(aux_res)
                aux_res = self.aux_task_head(aux_res)
                return x, aux_res
        return x
        
        