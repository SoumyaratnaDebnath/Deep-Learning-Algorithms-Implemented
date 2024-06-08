import torch
import torch.nn as nn

class Attention(nn.Module):
    """
    Attention mechanism.
    ---------------------
    Args:
        dim (int): input dimension. 
        n_heads (int): number of heads. 
        qkv_bias (bool): if True, add bias to qkv. 
        attn_p (float): dropout probability for attention. 
        proj_p (float): dropout probability for projection. 

    Attributes:
        scale (float): scaling factor. 
        qkv (nn.Linear): linear layer for qkv. 
        attn_drop (nn.Dropout): dropout layer for attention. 
        proj (nn.Linear): linear layer for projection. 
        proj_drop (nn.Dropout): dropout layer for projection. 
    """
    def __init__(self, dim, n_heads=8, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.scale = dim ** -0.5 # scaling factor
        self.dim = dim # input dimension
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # (dim, dim * 3) : q, k, v
        self.attn_drop = nn.Dropout(attn_p) # attention dropout
        self.proj = nn.Linear(dim, dim) # projection layer
        self.proj_drop = nn.Dropout(proj_p) # projection dropout

    def forward(self, x):
        """
        Forward pass of the attention layer. 
        -------------------------------------
        Args:
            x (torch.Tensor): input tensor. 
            shape : (n_samples, n_patches + 1, dim)

        Returns:
            torch.Tensor: output tensor. 
            shape : (n_samples, n_patches + 1, dim)
        """
        n_samples, n_patches, dim = x.shape

        if dim != self.dim:
            raise ValueError(f'Input dimension {dim} must match layer dimension {self.dim}.')
        
        qkv = self.qkv(x) # (n_samples, n_patches + 1, dim * 3)