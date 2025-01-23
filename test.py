import numpy as np
import torch

from model.SSSwinT import *

if __name__ == "__main__":

    seq_len = 256
    hidden_dim = 96
    layers = (2, 2, 6, 2)
    heads = (3, 6, 12, 24)
    channels = 4
    num_classes = 2
    head_dim = 32
    window_size = 8
    downscaling_factors=(4, 2, 2, 2)
    relative_pos_embedding=True
    abs_pos_embedding=True
    swin_ver=1    

    attn_drop=0. # Dropout ratio of attention weight.
    proj_drop=0. # Dropout ratio of output.            
    path_drop=0. # Drop prob of the MSA or the MLP blocks in Swin transformer block.
    mlp_drop=0. # Dropout ratio in the MLP in the Swin transformer block
    pos_drop=0. # Dropout ratio after the absolute positional embedding. 
    head_drop=0. # Dropout in the head    

    model = SSSwinT(
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        layers=layers,
        heads=heads,
        channels=channels,
        num_classes=num_classes,
        head_dim=head_dim, 
        window_size=window_size, # If len(signal) == 256, 7 - log_2 (w) = the number of stages. For example, if w = 4, we need 5 stages. So from the last stage, we will have window_size tokens represented by window_size scalar values.
        downscaling_factors=downscaling_factors,
        relative_pos_embedding=relative_pos_embedding,
        abs_pos_embedding=abs_pos_embedding,
        attn_drop=attn_drop,
        proj_drop=proj_drop,
        path_drop=path_drop,
        mlp_drop=mlp_drop,
        pos_drop=pos_drop,        
        swin_ver=swin_ver)    

    device = "cpu"

    BS = 2
    x = torch.rand(BS, seq_len, channels).to(device)

    model.to(device)

    out = model(x)    

    print(out)









