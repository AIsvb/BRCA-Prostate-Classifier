from src.models.feature_aggregation.assets.gigapath.model.LongNet import make_longnet_from_name
from functools import partial
import torch.nn as nn
import numpy as np
import torch

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def interpolate_pos_embed(model, checkpoint_model):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed

class PatchEmbed(nn.Module):
    """Slide Patch Embedding"""

    def __init__(
        self,
        in_chans=1536,
        embed_dim=768,
        norm_layer=None,
        bias=True,
    ):
        super().__init__()

        self.proj = nn.Linear(in_chans, embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, L, D = x.shape
        x = self.proj(x)
        x = self.norm(x)
        return x

class LongNetViT(nn.Module):
    """
    Backbone of Vision Transformer for downstream tasks

    Arguments:
    ----------
    embed_dim: int
        The number of input channels, should be the tile encoding dimension 1536.
    output_embed_dim: int
        The embedding dimension of the LongNet model.
    depth: int
        The number of LongNet layers in the LongNet model.
    slide_ngrids: int
        The number of grids in the slide.
    tile_size: int
        The tile size. Default is 256px.
    max_wsi_size: int
        The maximum size of the WSI.
    norm_layer: nn.LayerNorm
        The normalization layer used in the model.
    global_pool: bool
        Whether to use global pooling or not.
    dropout: float
        The dropout rate used in the model.
    drop_path_rate: float
        The drop path rate used in the model.
    n_classes: int
        The number of classes for classification
    """

    def __init__(self, 
                embed_dim=1536, 
                output_embed_dim=256, 
                depth=12, 
                slide_ngrids=1000, 
                tile_size=256,
                max_wsi_size=262144,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                global_pool=False, 
                dropout=0.25, 
                drop_path_rate=0.1, 
                n_classes=2,
                **kwargs):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(embed_dim, output_embed_dim)
        
        self.tile_size = tile_size
        self.slide_ngrids = slide_ngrids
        num_patches = slide_ngrids**2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, output_embed_dim))
        self.register_buffer('pos_embed', torch.zeros(1, num_patches + 1, output_embed_dim), persistent=False)  # fixed sin-cos embedding

        self.encoder_name = "LongNet_{}_layers_{}_dim".format(depth, output_embed_dim)
        if kwargs.get("mlp_ratio", 4.0) != 4.0:
            self.encoder_name += "_mlp{}".format(kwargs.get("mlp_ratio"))
        
        # get optimal segment length
        segment_length = self.get_optimal_segment_length(max_wsi_size, tile_size)
        self.encoder = make_longnet_from_name(self.encoder_name, drop_path_rate=drop_path_rate, dropout=dropout, segment_length=segment_length)
        self.norm = norm_layer(output_embed_dim)
        # --------------------------------------------------------------------------

        self.global_pool = global_pool
        print("Global Pooling:", self.global_pool)

        self.initialize_vit_weights()

        # --
        self.classifier = nn.Linear(output_embed_dim, n_classes)

    def initialize_vit_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.slide_ngrids, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def get_optimal_segment_length(self, max_wsi_size: int=262144, tile_size: int=256) -> str:
        '''
        Get the optimal segment length based on the maximum image size and tile size.
        
        Arguments:
        ----------
        max_wsi_size: int
            The maximum size of the WSI.
        tile_size: int
            The tile size.
        '''
        max_seq_len = (max_wsi_size // tile_size) ** 2
        # calculate the segment length
        segment_length = np.linspace(np.log2(1024), int(np.log2(max_seq_len)), 5)
        segment_length = np.power(2, segment_length).astype(int)
        # convert to str format
        segment_length = str(list(segment_length))
        return segment_length

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def coords_to_pos(self, coords, tile_size: int = 256):
        """
        This function is used to convert the coordinates to the positional indices

        Arguments:
        ----------
        coords: torch.Tensor
            The coordinates of the patches, of shape [N, L, 2]
        output: torch.Tensor
            The positional indices of the patches, of shape [N, L]
        """
        # -- added code: shift coords s.t. they are all positive and alligned with the x,y axes
        coords = coords - torch.min(coords, axis=1).values

        coords_ = torch.floor(torch.div(coords, tile_size))
        pos = torch.add(torch.mul(coords_[..., 0], self.slide_ngrids), coords_[..., 1])
        pos = torch.add(pos, 1) # add 1 for the cls token
        pos = pos.to(torch.long)

        return pos

    def forward(self, x, coords, all_layer_embed=False):
        """
        The forward pass of the model

        Arguments:
        ----------
        x: torch.Tensor
            The input tile embeddings, of shape [N, L, D]
        coords: torch.Tensor
            The coordinates of the patches, of shape [N, L, 2]
        all_layer_embed: bool
            Whether to return embeddings from all layers or not
        """
        # embed patches
        x = self.patch_embed(x)

        # get pos indices
        pos = self.coords_to_pos(coords, self.tile_size)  # [N, L]
        x = x + self.pos_embed[:, pos, :].squeeze(0)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        if all_layer_embed:
            output_ = self.encoder(src_tokens=None, token_embeddings=x, return_all_hiddens=all_layer_embed)
            x_list = output_["encoder_states"]
        else:
            output_ = self.encoder(src_tokens=None, token_embeddings=x)
            x_list = [output_["encoder_out"]]

        attn_raw = output_["attn_raw"]

        outcomes = []
        for x in x_list:
            if self.global_pool:
                x = x[:, 1:, :].mean(dim=1)  # global average pooling
                outcome = self.norm(x)
            else:
                x = self.norm(x)
                outcome = x[:, 0]
            outcomes.append(outcome)

        # --
        logits = self.classifier(outcomes[-1])
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = torch.nn.functional.softmax(logits, dim = 1)

        return logits, Y_prob, Y_hat, attn_raw, outcomes

def gigapath_slide_enc12l768d(**kwargs):
    model = LongNetViT(output_embed_dim=768, depth=12, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def gigapath_slide_enc24l1024d(**kwargs):
    model = LongNetViT(output_embed_dim=1024, depth=24, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def gigapath_slide_enc12l1536d(**kwargs):
    model = LongNetViT(output_embed_dim=1536, depth=12, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
