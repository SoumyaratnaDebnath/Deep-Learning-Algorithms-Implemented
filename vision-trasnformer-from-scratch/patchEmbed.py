import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """
    Split image into patches and then embed them.   
    ----------------------------------------------
    Args:
        img_size (int): input image size. 
        patch_size (int): patch size. 
        in_chans (int): number of input channels. 
        embed_dim (int): number of embedding dimensions. 

    Attributes:
        n_patches (int): number of patches inside an image. 
        proj (nn.Conv2d): convolutional layer for embedding. 
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        Forward pass of the patch embedding. 
        -------------------------------------
        Args:
            x (torch.Tensor): input tensor. 
            shape : (n_samples, in_chans, img_size, img_size)

        Returns:
            torch.Tensor: flattened tensor. 
            shape : (n_samples, n_patches, embed_dim)
        """
        x = self.proj(x) # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2) # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2) # (n_samples, n_patches, embed_dim)

        return x