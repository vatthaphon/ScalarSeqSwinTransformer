import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange
from timm.models.layers import trunc_normal_ as tm_trunc_normal_
from timm.models.layers import DropPath

class CyclicShift(nn.Module): # Checked
    def __init__(self, displacement):
        super().__init__()

        self.displacement = displacement # negative == the left-most token becomes the right-most token.

    def forward(self, x):
        '''
        x: b l c
        '''     
        return torch.roll(x, shifts=self.displacement, dims=1) # Shift the length dimension.

def create_mask(window_size, displacement): # Checked
    '''
    Only operate on the right-most attention window.
    '''

    assert displacement > 0, f"create_mask: displacement {displacement} is always positive."

    mask = torch.zeros(window_size, window_size)

    mask[-displacement:, :-displacement] = float('-inf') # Deal with the last displacement rows.
    mask[:-displacement, -displacement:] = float('-inf') # Deal with the last displacement cols.        
    
    return mask

def get_relative_distances(window_size):

    row_i = np.array([[row for _ in range(window_size)] for row in range(window_size)]) # Create a matrix, of which element is the row index.
    col_i = np.array([[col for col in range(window_size)] for _ in range(window_size)]) # Create a matrix, of which element is the col index.

    return torch.tensor(col_i - row_i)        

class WindowAttention(nn.Module):
    def __init__(self, 
        dim, # the embeded size, aka channels.
        heads, # the number of heads.
        head_dim, # head_dim * heads == dim.
        shifted, 
        window_size, 
        relative_pos_embedding,
        attn_drop=0., # Dropout ratio of attention weight.
        proj_drop=0., # Dropout ratio of output.
        swin_ver=1):

        super().__init__()

        assert head_dim * heads == dim, f"WindowAttention: the embeded size {dim} should equal {head_dim} * {heads}"

        self.swin_ver = swin_ver            
        self.heads = heads
        self.scale = head_dim ** 0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted # W-MSA or SW-MSA? 

        inner_dim = head_dim*heads

        if self.shifted: # Shift to the right by window_size // 2 samples. The left-most window_size // 2 samples will reappear to the right because it is a cyclic shift.
            displacement = window_size // 2 
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)

            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement), requires_grad=False) # mask is not learnable.                

        self.to_qkv = nn.Linear(dim, inner_dim*3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1 # self.relative_indices.shape = (window_size, window_size)
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))

        self.to_out = nn.Linear(inner_dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)            

        match self.swin_ver:
            case 1: # Swin transformer V1
                pass
            case 2: # Swin transformer V2
                self.tau = nn.Parameter(torch.tensor(0.01), requires_grad=True)
            case _:
                raise Exception("Wrong swin version.")

    def forward(self, x):
        '''
        x: b l c
        '''            
        if self.shifted:
            x = self.cyclic_shift(x)

        b, n, c, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1) # Split the tensor into chunks along the last dimension.

        nw = n // self.window_size # The number of windows that we have. nw == 1 is possible.

        q, k, v = map(lambda t: rearrange(t, 'b (nw w) (h d) -> b h nw w d', h=h, w=self.window_size), qkv) # d=head_dim. q.shape = (batches, #heads, #windows, windows_size, head_dim)

        match self.swin_ver:
            case 1: # Swin transformer V1 using the dot product similarity.
                dots = einsum('b h w i d, b h w j d -> b h w i j', q, k)/self.scale # dots = (b, heads, #windows, window_size, window_size). i and j now become windows_size. einsum doesn't allow nw, I then change to w.
            case 2: # Swin transformer V2 using the cosine similarity.
                q = F.normalize(q, p=2, dim=-1) # Normalize along the head_dim wrt L2 norm.
                k = F.normalize(k, p=2, dim=-1) # Normalize along the head_dim wrt L2 norm.

                dots = einsum('b h w i d, b h w j d -> b h w i j', q, k)/self.tau # i and j now become windows_size. einsum doesn't allow nw, I then change to w. dots.shape == (b, #heads, #windows, windows_size, windows_size)
            case _:
                raise Exception("Wrong swin version.")
                
        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices, self.relative_indices] # dots.shape == (b, #heads, #windows, windows_size, windows_size). pos_embedding.shape == (windows_size, windows_size).

        if self.shifted: ##############
            dots[:, :, -1:] += self.left_right_mask # Add mask only to the right-most window. dots.shape == (b, h, nw, windows_size, windows_size). We focus only the last window because the displacement is half of the windows_size.

        attn = dots.softmax(dim=-1) # attn.shape == (b, #heads, #windows, windows_size, windows_size)

        attn = self.attn_drop(attn)

        ## out.shape == (batches, #heads, #windows, windows_size, head_dim)
        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v) # d=head_dim. attn.shape == (b, #heads, #windows, windows_size, windows_size). v.shape = (batches, #heads, #windows, windows_size, head_dim)

        out = rearrange(out, 'b h nw w d -> b (nw w) (h d)', h=h, w=self.window_size, nw=nw) # out.shape == x.shape == (b, #tokens, emb_size)

        out = self.to_out(out)

        out = self.proj_drop(out)

        if self.shifted:
            x = self.cyclic_back_shift(x)

        return out 

class FeedForward(nn.Module):
    def __init__(self, 
        dim, # out_channels.
        mlp_dim,
        mlp_drop):
        super().__init__()

        self.net = nn.Sequential(
            ## First layer is a non-linear layer because of GELU.
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(mlp_drop),

            ## Second layer is a linear layer.
            nn.Linear(mlp_dim, dim),
            nn.Dropout(mlp_drop),
        )

    def forward(self, x):
        return self.net(x)

class SwinBlock(nn.Module):
    def __init__(self, 
        dim, # out_channels
        heads, 
        head_dim, 
        mlp_dim, 
        shifted, 
        window_size, 
        relative_pos_embedding,
        attn_drop=0., # Dropout ratio of attention weight.
        proj_drop=0., # Dropout ratio of output.            
        path_drop=0., # Stochastic depth rate.
        mlp_drop=0.,
        swin_ver=1):
        super().__init__()

        self.swin_ver = swin_ver

        self.att_norm = nn.LayerNorm(dim) # Normalize over the last dimension, i.e. channels.
        self.att = WindowAttention(dim=dim, heads=heads, head_dim=head_dim, shifted=shifted, window_size=window_size, relative_pos_embedding=relative_pos_embedding, attn_drop=attn_drop, proj_drop=proj_drop, swin_ver=swin_ver)        

        self.mlp_norm = nn.LayerNorm(dim) # Normalize over the last dimension, i.e. channels.
        self.mlp = FeedForward(dim=dim, mlp_dim=mlp_dim, mlp_drop=mlp_drop)

        self.path_drop = DropPath(path_drop) if path_drop > 0. else nn.Identity()

    def forward(self, x):
    
        match self.swin_ver:
            case 1: # Swin transformer V1
                x = x + self.path_drop(self.att(self.att_norm(x)))
                x = x + self.path_drop(self.mlp(self.mlp_norm(x)))
            case 2: # Swin transformer V2
                x = x + self.path_drop(self.att_norm(self.att(x)))
                x = x + self.path_drop(self.mlp_norm(self.mlp(x)))
            case _:
                raise Exception("Incorrect swin's version.")

        return x

class PatchMerging(nn.Module):
    def __init__(self, 
        seq_len,
        in_channels, 
        out_channels, 
        downscaling_factor):
        super().__init__()    

        assert seq_len % downscaling_factor == 0, f"Sequence length {seq_len} has to be a multiple of {downscaling_factor}."

        self.out_seq_len = seq_len // downscaling_factor

        ## Original implementation was fixed to decreasing of the sequence's length by half. So the number of required stages is fixed to reach desired sequence's length.
        ## Therefore, I prefer to use Conv1D to give more room for adjusting the number of stages.
        self.patch_merge = nn.Conv1d(in_channels, out_channels, kernel_size=downscaling_factor, stride=downscaling_factor) # b in_c l -> b out_c l

    def forward(self, x):
        '''
        x: b c l
        '''
        x = self.patch_merge(x)
        x = x.transpose(2, 1) # b l c

        return x

class StageModule(nn.Module):
    def __init__(self,
        stage,
        seq_len,
        in_channels,
        hidden_dimension, # basically, it is out_channels.
        layers, 
        downscaling_factor, 
        num_heads, 
        head_dim, 
        window_size,
        relative_pos_embedding,
        abs_pos_embedding=False, # Absolute positional embedding, applied before the first stage.
        norm_layer=None, # Norm layer after patch merging.
        attn_drop=0., 
        proj_drop=0., 
        path_drop=0., 
        mlp_drop=0., 
        pos_drop=0.,
        swin_ver=1):
        super().__init__()

        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.stage = stage

        self.window_size = window_size
        self.abs_pos_embedding = abs_pos_embedding

        self.patch_merging = PatchMerging(seq_len=seq_len, in_channels=in_channels, out_channels=hidden_dimension, downscaling_factor=downscaling_factor)

        self.out_seq_len = self.patch_merging.out_seq_len
        self.out_channels = hidden_dimension

        if self.abs_pos_embedding:
            self.pos_embedding = nn.Parameter(torch.zeros(1, self.out_seq_len, hidden_dimension)) # I included the 1 in the first dimention to explicitly indicate that all batches share the same embedding.
            tm_trunc_normal_(self.pos_embedding, std=.02) # THis is based on the original Swin implmentation.        

        self.norm = norm_layer(hidden_dimension) if norm_layer is not None else None

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, 
                heads=num_heads, 
                head_dim=head_dim, 
                mlp_dim=hidden_dimension * 4, 
                shifted=False, 
                window_size=window_size, 
                relative_pos_embedding=relative_pos_embedding, 
                attn_drop=attn_drop, 
                proj_drop=proj_drop, 
                path_drop=path_drop, 
                mlp_drop=mlp_drop, 
                swin_ver=swin_ver), # W-MSA
                
                SwinBlock(dim=hidden_dimension, 
                heads=num_heads, 
                head_dim=head_dim, 
                mlp_dim=hidden_dimension * 4, 
                shifted=True, 
                window_size=window_size, 
                relative_pos_embedding=relative_pos_embedding, 
                attn_drop=attn_drop, 
                proj_drop=proj_drop, 
                path_drop=path_drop, 
                mlp_drop=mlp_drop, 
                swin_ver=swin_ver), # SW-MSA
            ]))

        self.pos_drop = nn.Dropout(pos_drop) # Positional dropout.

    def forward(self, x):
        '''
        x: b c l
        '''

        x = self.patch_merging(x) # b l c

        if self.norm is not None: # Only active in the initial stage. So, non-initial stages will have norm_layer == None.
            x = self.norm(x)  

        if self.abs_pos_embedding: # Only active in the initial stage. So, non-initial stages will have abs_pos_embedding == False.

            assert (x.shape[1] == self.pos_embedding.shape[1]) and (x.shape[2] == self.pos_embedding.shape[2])

            x = x + self.pos_embedding # Use learned positional embedding instead of the sin+cosine positional embedding like in the vanilla Transformer.

        x = self.pos_drop(x) # Only active in the initial stage. So, non-initial stages will have pos_drop == 0.

        for regular_block, shifted_block in self.layers:
            x = regular_block(x) # b l c
            x = shifted_block(x) # b l c

        x = x.transpose(2, 1) # b c l

        return x

class SSSwinT(nn.Module):
    def __init__(self,
        *, # Parameters after * must be passed as a keyword. 
        seq_len, # Length of the input sequence.
        hidden_dim, # the initial embeded size.
        layers, # The number of Swin Transformer Block in each stage.
        heads, 
        channels=3, # RGB
        num_classes=1000, 
        head_dim=32, 
        window_size=7, # len(signal) / (2**(M+1)) == window_size, where M is the number of stages. That is, the length of the signal from the last stage should equal window_size.
        downscaling_factors=(4, 2, 2, 2), # Liu2021: len(seq)[i] == len(seq)[i-1]/downscaling_factors[i], i=1,...,M and len(seq)[i-1] is the original length. That is, the length of the signal shrinks by how many factor.
        relative_pos_embedding=True,
        abs_pos_embedding=False,
        attn_drop=0., # Dropout ratio of attention weight.
        proj_drop=0., # Dropout ratio of output.            
        path_drop=0., # Stochastic depth rate.
        mlp_drop=0.,
        pos_drop=0.,
        head_drop=0.,
        swin_ver=1):
        super().__init__()

        ## Add Swin Transformer Block.
        self.stages = nn.ModuleList() # Cannot use Sequential because it is immutable.

        print(f"Shape of initial input: (l={seq_len}, c={channels})")

        ## For the first stage, we replace Patch Partition and Linear Embedding by Patch Merging.
        self.stages.append(StageModule(stage=0,
            seq_len=seq_len, 
            in_channels=channels, 
            hidden_dimension=hidden_dim, 
            layers=layers[0], 
            downscaling_factor=downscaling_factors[0], 
            num_heads=heads[0], 
            head_dim=head_dim, 
            window_size=window_size, 
            relative_pos_embedding=relative_pos_embedding, 
            abs_pos_embedding=abs_pos_embedding, 
            norm_layer=nn.LayerNorm,                 
            attn_drop=attn_drop, 
            proj_drop=proj_drop, 
            path_drop=path_drop, 
            mlp_drop=mlp_drop, 
            pos_drop=pos_drop, 
            swin_ver=swin_ver))
        seq_len = self.stages[-1].out_seq_len            

        print(f"Shape of the input after stage 0: (l={seq_len}, c={self.stages[-1].out_channels})")

        in_channels = hidden_dim
        hidden_dimension = hidden_dim*2
        for i in range(1, len(layers)):
            self.stages.append(StageModule(stage=i,
                seq_len=seq_len, 
                in_channels=in_channels, 
                hidden_dimension=hidden_dimension, 
                layers=layers[i], 
                downscaling_factor=downscaling_factors[i], 
                num_heads=heads[i], 
                head_dim=head_dim, 
                window_size=window_size, 
                relative_pos_embedding=relative_pos_embedding, 
                attn_drop=attn_drop, 
                proj_drop=proj_drop, 
                path_drop=path_drop, 
                mlp_drop=mlp_drop, 
                swin_ver=swin_ver))
            seq_len = self.stages[-1].out_seq_len

            print(f"Shape of the input after stage {i}: (l={seq_len}, c={self.stages[-1].out_channels})")

            in_channels *= 2
            hidden_dimension *= 2

        self.norm = nn.LayerNorm(self.stages[-1].out_channels)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.head_drop = nn.Dropout(head_drop)
        self.head = nn.Linear(self.stages[-1].out_channels, num_classes)

    def forward_features(self, x):
        '''
        x: b c l
        '''

        for stage in self.stages:
            x = stage(x) # x: b c l

        x = self.norm(x.transpose(2, 1)) # b l c

        x = self.avgpool(x.transpose(2, 1)) # b c 1
        
        x = torch.flatten(x, 1) # b c         

        return x

    def forward_head(self, x):
        '''
        x: b c
        '''

        x = self.head_drop(x) # b c

        x = self.head(x) # b classes

        return x

    def forward(self, x):
        '''
        x: b l c
        '''
        x = x.transpose(2, 1) # b c l

        x = self.forward_features(x) # b c

        x = self.forward_head(x) # b classes  

        return x