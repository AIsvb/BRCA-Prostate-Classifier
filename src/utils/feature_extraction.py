import numpy as np
import torch


class UnfoldCollator:
    def __init__(self, shape, stride):
        """
        Initializes the collate function with the specified shape and stride.
        
        Args:
            shape (int): The size of the square tile to unfold the image into.
            stride (int): The stride with which to slide the window for unfolding.
        """
        self.shape = shape
        self.stride = stride
        self.unfold = torch.nn.Unfold(kernel_size=shape, stride=stride)

    def __call__(self, imgs_tensor, coords):
        bs, c, _, _ = imgs_tensor.shape # Shape: (num_4k_tiles, channels, height, width)

        # Use torch.nn.Unfold to unfold the images into tiles
        unfolded_imgs = self.unfold(imgs_tensor)  # Shape: (num_4k_tiles, C * shape * shape, num_sub_tiles)
        unfolded_imgs = unfolded_imgs.view(bs, c, self.shape, self.shape, -1) # Shape: (num_4k_tiles, channels, shape, shape, num_sub_tiles)
        unfolded_imgs = unfolded_imgs.permute(0, 4, 1, 2, 3).reshape(-1, c, self.shape, self.shape) # Shape: (num_4k_tiles * num_sub_tiles, channels, shape, shape)

        # Calculate the unfolded coordinates for each tile
        # Generate the grid of top-left coordinates based on the original coords and stride
        img_height, img_width = imgs_tensor.shape[2], imgs_tensor.shape[3]

        # Coordinates for a single image
        coords_x = np.arange(0, img_width - self.shape + 1, self.stride)
        coords_y = np.arange(0, img_height - self.shape + 1, self.stride)
        grid_x, grid_y = np.meshgrid(coords_x, coords_y)

        # Stack the coordinates grid into pairs of (x, y)
        grid_coords = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)  # Shape: (num_sub_tiles, 2)

        # Broadcast grid coordinates to the whole batch and add the base coordinates
        all_coords = (grid_coords[None, :, :] + coords[:, None, :]).reshape(-1, 2)  # Shape: (total_tiles, 2)

        # Return the unfolded images and their new coordinates
        return unfolded_imgs, all_coords # torch.Tensor | np.array

# To load from .pt: x = torch.load(file.pt); x = x.reshape(-1, 256, embed_dim); x.unfold(1, 16, 16).transpose(1,2) --> [bs, embed_dim, 16, 16] which is correct input for HIPT4k
