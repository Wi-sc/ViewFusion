import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
import math
import numpy as np
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
)
from ldm.modules.attention import SpatialTransformer
from ldm.util import instantiate_from_config, exists
from .brdf_renderer import Renderer

# dummy replace
def convert_module_to_f16(x):
    pass

def convert_module_to_f32(x):
    pass

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class TransposedUpsample(nn.Module):
    'Learned 2x upsampling without padding'
    def __init__(self, channels, out_channels=None, ks=5):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.up = nn.ConvTranspose2d(self.channels,self.out_channels,kernel_size=ks,stride=2)

    def forward(self,x):
        return self.up(x)


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)
    
class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        # self.emb_layers = nn.Sequential(
        #     nn.SiLU(),
        #     linear(
        #         emb_channels,
        #         2 * self.out_channels if use_scale_shift_norm else self.out_channels,
        #     ),
        # )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x,), self.parameters(), self.use_checkpoint
        )


    def _forward(self, x):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        # emb_out = self.emb_layers(emb).type(h.dtype)
        # while len(emb_out.shape) < len(h.shape):
        #     emb_out = emb_out[..., None]
        # if self.use_scale_shift_norm:
        #     out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        #     scale, shift = torch.chunk(emb_out, 2, dim=1)
        #     h = out_norm(h) * (1 + scale) + shift
        #     h = out_rest(h)
        # else:
        #     h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        #return pt_checkpoint(self._forward, x)  # pytorch

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)
    
class UNet(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None
    ):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        #self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")  # todo: convert to warning

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        # time_embed_dim = model_channels * 4
        # self.time_embed = nn.Sequential(
        #     linear(model_channels, time_embed_dim),
        #     nn.SiLU(),
        #     linear(time_embed_dim, time_embed_dim),
        # )

        # if self.num_classes is not None:
        #     self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList([conv_nd(dims, in_channels, model_channels, 3, padding=1)])

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    # if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                    #     layers.append(
                    #         AttentionBlock(
                    #             ch,
                    #             use_checkpoint=use_checkpoint,
                    #             num_heads=num_heads,
                    #             num_head_channels=dim_head,
                    #             use_new_attention_order=use_new_attention_order,
                    #         ) if not use_spatial_transformer else SpatialTransformer(
                    #             ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                    #             disable_self_attn=disabled_sa
                    #         )
                    #     )
                self.input_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    nn.Sequential(
                        ResBlock(
                            ch,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = nn.Sequential(
            ResBlock(
                ch,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            # AttentionBlock(
            #     ch,
            #     use_checkpoint=use_checkpoint,
            #     num_heads=num_heads,
            #     num_head_channels=dim_head,
            #     use_new_attention_order=use_new_attention_order,
            # ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
            #                 ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
            #             ),
            ResBlock(
                ch,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    # if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                    #     layers.append(
                    #         AttentionBlock(
                    #             ch,
                    #             use_checkpoint=use_checkpoint,
                    #             num_heads=num_heads_upsample,
                    #             num_head_channels=dim_head,
                    #             use_new_attention_order=use_new_attention_order,
                    #         ) if not use_spatial_transformer else SpatialTransformer(
                    #             ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                    #             disable_self_attn=disabled_sa
                    #         )
                    #     )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x):
        hs = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h)
            hs.append(h)
        h = self.middle_block(h)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h)
        h = h.type(x.dtype)
        return self.out(h)
        
class RGB2BRDF(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 decoder_config,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image_target",
                 colorize_nlabels=None,
                 monitor=None,
                 fov=49.1, 
                 gamma=0.8, 
                 attenuation=True
                 ):
        super().__init__()
        self.image_key = image_key
        self.unet = UNet(**ddconfig)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, decoder_config["z_channels"], 1)
        self.decoder = Decoder(**decoder_config)
        self.renderer = Renderer(fov, gamma, attenuation)
        # assert ddconfig["double_z"]

        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys) 

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def forward(self, x):
        return self.unet(x)

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec
    
    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        # x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    # def training_step(self, batch, batch_idx, optimizer_idx):
    def training_step(self, batch, batch_idx):
        rgb_feature = self.get_input(batch, "rgb")
        albedo_feature = self.get_input(batch, "albedo")
        normal_feature = self.get_input(batch, "normal")
        roughness_feature = self.get_input(batch, "roughness")
        specular_feature = self.get_input(batch, "specular")
        gt_feature = torch.cat([albedo_feature, normal_feature, roughness_feature, specular_feature], dim=1)
        
        brdf_feature_maps = self(rgb_feature)
        brdf_reconstruction_loss = torch.abs(gt_feature-brdf_feature_maps)
        brdf_reconstruction_loss = torch.sum(brdf_reconstruction_loss) / brdf_reconstruction_loss.shape[0]

        self.log("train/brdf_rec_loss", brdf_reconstruction_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        log_dict = {}
        log_dict["train_brdf_rec_loss"] = brdf_reconstruction_loss
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return brdf_reconstruction_loss

    def validation_step(self, batch, batch_idx):
        rgb_feature = self.get_input(batch, "rgb")
        albedo_feature = self.get_input(batch, "albedo")
        normal_feature = self.get_input(batch, "normal")
        roughness_feature = self.get_input(batch, "roughness")
        specular_feature = self.get_input(batch, "specular")
        gt_feature = torch.cat([albedo_feature, normal_feature, roughness_feature, specular_feature], dim=1)

        brdf_feature_maps = self(rgb_feature)
        brdf_reconstruction_loss = torch.abs(gt_feature-brdf_feature_maps)
        brdf_reconstruction_loss = torch.sum(brdf_reconstruction_loss) / brdf_reconstruction_loss.shape[0]

        self.log("val/rec_loss", brdf_reconstruction_loss)
        self.log_dict({"val/rec_loss":brdf_reconstruction_loss})
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        # opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
        #                           list(self.decoder.parameters())+
        #                           list(self.quant_conv.parameters())+
        #                           list(self.post_quant_conv.parameters()),
        #                           lr=lr, betas=(0.5, 0.9))
        opt_ae = torch.optim.Adam(list(self.unet.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return [opt_ae], []
        # opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
        #                             lr=lr, betas=(0.5, 0.9))
        # return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        rgb_feature = self.get_input(batch, "rgb").to(self.device)
        albedo_feature = self.get_input(batch, "albedo").to(self.device)
        normal_feature = self.get_input(batch, "normal").to(self.device)
        roughness_feature = self.get_input(batch, "roughness").to(self.device)
        specular_feature = self.get_input(batch, "specular").to(self.device)

        brdf_feature_maps = self(rgb_feature)
        brdf_feature_maps = brdf_feature_maps/0.18215
        if not only_inputs:
            # batch_size, _, res1, res2 = rgb_feature.shape
            # brdf_feature_maps = brdf_feature_maps.reshape(batch_size*4, -1, res1, res2)
            # batch_id = torch.arange(batch_size)*4

            albedo_maps = self.decode(brdf_feature_maps[:, :4, ...])
            normal_maps = self.decode(brdf_feature_maps[:, 4:8, ...])
            roughness_maps = self.decode(brdf_feature_maps[:, 8:12, ...])
            specular_maps = self.decode(brdf_feature_maps[:, 12:16, ...])

            brdf_maps = {
                    'diffuse': (albedo_maps/2+0.5).clamp(0.01, 0.99),
                    'normal': (normal_maps).clamp(-1, 1),
                    'roughness': (roughness_maps/2+0.5).mean(dim=1, keepdim=True).clamp(0.01, 0.99),
                    'specular': (specular_maps/2+0.5).mean(dim=1, keepdim=True).clamp(0.01, 0.99),
                }
            rgb_reconstructions = self.renderer(brdf_maps)
            rgb_reconstructions = rgb_reconstructions * 2 - 1
            log["reconstructions"] = rgb_reconstructions
            log["albedo"] = albedo_maps
            log["normal"] = normal_maps
            log["roughness"] = roughness_maps
            log["specular"] = specular_maps

            gt_albedo_maps = self.decode(albedo_feature/0.18215)
            gt_normal_maps = self.decode(normal_feature/0.18215)
            gt_roughness_maps = self.decode(roughness_feature/0.18215)
            gt_specular_maps = self.decode(specular_feature/0.18215)
            # log["inputs"] = rgb_inputs
            # log["brdf_gt"] = rgb_brdf
            log["albedo_gt"] = gt_albedo_maps
            log["normal_gt"] = gt_normal_maps
            log["roughness_gt"] = gt_roughness_maps
            log["specular_gt"] = gt_specular_maps
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

