from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Sequence
from torchvision.transforms.functional import gaussian_blur
from monai.networks.blocks.dynunet_block import UnetOutBlock, get_conv_layer
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from scripts_formal.utils.elements_vit import ModifiedViTEncoder

class VolumeResidualBlock3D(nn.Module):
    """Residual 3D convolution block used inside the trainable z-fusion adapter."""
    def __init__(self, channels: int, norm_groups: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine 3D features while preserving their channel and spatial shape."""
        return self.activation(x + self.block(x))


class TrainableVolumeToSamAdapter(nn.Module):
    """Convert fixed-depth input volumes to 2D SAM-compatible images with spatial z attention."""
    def __init__(self,
                 in_channels: int,
                 input_z_size: int = 10,
                 adapter_channels: int = 32,
                 res_blocks: int = 32,
                 norm: str = "groupnorm") -> None:
        super().__init__()
        if norm.lower() != "groupnorm":
            raise ValueError("Only groupnorm is supported for the volume adapter.")
        self.in_channels = in_channels
        self.input_z_size = input_z_size
        self.adapter_channels = adapter_channels
        self.res_blocks = res_blocks
        norm_groups = self._resolve_group_count(adapter_channels)

        self.channel_adapter = nn.Conv3d(in_channels, adapter_channels, kernel_size=1)
        self.blocks = nn.Sequential(*[
            VolumeResidualBlock3D(adapter_channels, norm_groups) for _ in range(res_blocks)
        ])
        self.spatial_z_attention = nn.Conv3d(adapter_channels, 1, kernel_size=1)
        self.output_adapter = nn.Conv2d(adapter_channels, 1, kernel_size=1)
        self.last_attention: torch.Tensor | None = None

    @staticmethod
    def _resolve_group_count(channels: int) -> int:
        """Choose the largest standard GroupNorm group count that divides channels."""
        for group_count in (8, 4, 2, 1):
            if channels % group_count == 0:
                return group_count
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse (B, C, Z, H, W) volumes into (B, 1, H, W) grayscale SAM skip inputs."""
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {x.shape[1]}.")
        if x.shape[2] != self.input_z_size:
            raise ValueError(
                f"Expected Z={self.input_z_size} from dataset resampling, got Z={x.shape[2]}."
            )

        feat = self.blocks(self.channel_adapter(x))
        attention = torch.softmax(self.spatial_z_attention(feat), dim=2)
        self.last_attention = attention.detach()
        fused_feat = (feat * attention).sum(dim=2)
        return self.output_adapter(fused_feat)

class SamUNetR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """
    def __init__(
        self,
        vit_encoder: ModifiedViTEncoder|None = None,
        hidden_size: int = 1024, # equal to the encoder_embed_dim in SAM
        hidden_layer_hooks: list = [6, 12, 18, 24],
        feature_size: int = 16,
        norm_name: tuple | str = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims: int = 2,
        num_class: int = 1,
        use_prediction_head: bool = False,
        prediction_head_channels: int|None = None,
        prediction_head_blocks: int = 3,
        input_z_size: int = 10,
        image_in_channels: int = 1,
        volume_adapter_type: str = "spatial_z_attention",
        volume_adapter_channels: int = 8,
        volume_adapter_res_blocks: int = 4,
        volume_adapter_norm: str = "groupnorm",
        train_volume_adapter: bool = True,
        vit_adapt: str = "none",
        lora_rank: int = 4,
        lora_layers: Sequence[int] | None = None,
        use_image_skip: bool = True,
        sam_skip_channels: int = 1,
        **kwargs,
    ) -> None:
        """
        Args:
            out_channels (int): Number of output channels.
            img_size (int or tuple): Size of the input image.
            feature_size (int): Feature size for the UNETR model.
            hidden_size (int): Hidden size for the transformer encoder.
            mlp_dim (int): Dimension of the MLP in the transformer encoder.
            num_heads (int): Number of attention heads in the transformer encoder.
            pos_embed (str): Type of positional embedding to use ('conv' or 'perceptron').
            norm_name (str or tuple): Normalization type to use.
            conv_block (bool): Whether to use convolutional blocks.
            res_block (bool): Whether to use residual blocks.
            spatial_dims (int): Number of spatial dimensions (2 or 3).
            use_prediction_head (bool): Whether to use thicker prediction heads for the hv/mask branch.
            prediction_head_channels (int or None): Hidden channels in each thicker hv/mask prediction head.
            prediction_head_blocks (int): Number of convolutional blocks in each thicker hv/mask prediction head.
            input_z_size (int): Fixed z size used by input volumes.
            image_in_channels (int): Number of selected biological channels supplied to the volume adapter.
            volume_adapter_type (str): Volume adapter implementation name.
            volume_adapter_channels (int): Hidden feature channels inside the volume adapter.
            volume_adapter_res_blocks (int): Number of 3D residual blocks in the volume adapter.
            volume_adapter_norm (str): Normalization type inside the volume adapter.
            train_volume_adapter (bool): Whether volume adapter parameters are trainable.
            vit_adapt (str): ViT adaptation mode: ``none`` (frozen backbone) or ``lora`` (SAMed-style Q/V LoRA).
            lora_rank (int): LoRA rank ``r`` when ``vit_adapt='lora'``.
            lora_layers (Sequence[int] | None): Block indices to inject LoRA; ``None`` means all blocks.
            use_image_skip (bool): If True, decoder2 uses UnetrUpBlock with encoder1 image skip;
                if False, decoder2 uses UnetrUpBlockNoSkip (ViT tokens only).
            sam_skip_channels (int): Channel count of the grayscale image fed into encoder1 when
                ``use_image_skip=True``.
            **kwargs: Accepted for backward compatibility.

        Examples::

            >>> model = UNETR(in_channels=1, out_channels=3, img_size=(128, 128, 128))
            >>> input_tensor = torch.randn(1, 1, 128, 128, 128)
            >>> output = model(input_tensor)

        """

        super().__init__()


        self.num_layers = 12
        self.patch_size = (16, 16)
        self.feat_size = (64, 64)
        self.classification = False
        self.hidden_size = hidden_size
        self.hidden_layer_hooks = hidden_layer_hooks
        self.input_z_size = input_z_size
        self.volume_adapter_type = volume_adapter_type
        self.volume_adapter_channels = volume_adapter_channels
        self.volume_adapter_res_blocks = volume_adapter_res_blocks
        self.volume_adapter_norm = volume_adapter_norm
        self.train_volume_adapter = train_volume_adapter
        self.use_image_skip = use_image_skip
        self.sam_skip_channels = sam_skip_channels
        self.volume_adapter: TrainableVolumeToSamAdapter | None = None
        self.vit_adapt = vit_adapt
        self.lora_rank = lora_rank
        self.lora_layers = list(lora_layers) if lora_layers is not None else None
        self.lora_modules: list[LoRAQKV] = []

        self.vit = vit_encoder
        self.norm = nn.LayerNorm(self.hidden_size)

        '''
        Parameters of the SAM large model
        '''
        prompt_embed_dim = 256
        image_size = 1024
        vit_patch_size = 16
        encoder_embed_dim = self.hidden_size,
        encoder_depth = 24,
        encoder_num_heads = 16,
        encoder_global_attn_indexes = [5, 11, 17, 23]

        if self.vit is None:
            '''
            Initialize the Model. This will only be accessed when the vit_encoder is None.
            '''
            self.vit = ModifiedViTEncoder(
                depth=encoder_depth,
                embed_dim=encoder_embed_dim,
                img_size=image_size,
                mlp_ratio=4,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                num_heads=encoder_num_heads,
                patch_size=vit_patch_size,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=encoder_global_attn_indexes,
                window_size=14,
                out_chans=prompt_embed_dim,
            )
            self.vit.strip_neck()
        elif hasattr(self.vit, 'strip_neck'):
            self.vit.strip_neck()
        elif hasattr(self.vit, 'neck'):
            delattr(self.vit, 'neck')

        # Freeze SAM ViT backbone; optionally inject SAMed-style LoRA on Q/V.
        self.vit.requires_grad_(False)
        if self.vit_adapt == "lora":
            self.lora_modules = inject_lora_into_vit(
                self.vit, rank=self.lora_rank, lora_layers=self.lora_layers)
        elif self.vit_adapt != "none":
            raise ValueError(f"Unsupported vit_adapt={self.vit_adapt!r}; expected 'none' or 'lora'.")
        self.configure_input_adapter(image_in_channels=image_in_channels, force=True)

        self.upsampler = nn.Module()
        self.upsampler.hv_mask_decoder = UNETR_UpSampler(
            out_channels=None,
            hidden_size=self.hidden_size,
            feature_size=feature_size,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
            spatial_dims=spatial_dims,
            use_image_skip=use_image_skip,
            in_channels=sam_skip_channels)
        self.upsampler.cyto_type_decoder = UNETR_UpSampler(
            out_channels=None,
            hidden_size=self.hidden_size,
            feature_size=feature_size,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
            spatial_dims=spatial_dims,
            use_image_skip=use_image_skip,
            in_channels=sam_skip_channels)

        if use_prediction_head:
            prediction_hidden_channels = prediction_head_channels or feature_size * 2
            self.upsampler.nuc_mask_predictor = PredictionHead(
                spatial_dims=spatial_dims,
                in_channels=feature_size,
                hidden_channels=prediction_hidden_channels,
                block_num=prediction_head_blocks,
                norm_name=norm_name,
                res_block=res_block,
                out_channels=1)
            self.upsampler.nuc_hv_predictor = PredictionHead(
                spatial_dims=spatial_dims,
                in_channels=feature_size,
                hidden_channels=prediction_hidden_channels,
                block_num=prediction_head_blocks,
                norm_name=norm_name,
                res_block=res_block,
                out_channels=2)
        else:
            self.upsampler.nuc_mask_predictor = UnetOutBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size,
                out_channels=1)
            self.upsampler.nuc_hv_predictor = UnetOutBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size,
                out_channels=2)
        self.upsampler.cyto_type_predictor = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=num_class)

    @property
    def uses_volume_adapter(self) -> bool:
        """True when Zt > 1 and the trainable 3D volume adapter is active."""
        return self.input_z_size > 1

    def configure_input_adapter(self, image_in_channels: int, force: bool = False) -> None:
        """Configure the volume adapter for Zt > 1; skip adapter entirely when Zt == 1."""
        already_configured = (
            getattr(self, 'image_in_channels', None) == image_in_channels
            and (
                (not self.uses_volume_adapter and self.volume_adapter is None)
                or (
                    self.volume_adapter is not None
                    and self.volume_adapter.in_channels == image_in_channels
                    and self.volume_adapter.input_z_size == self.input_z_size
                )
            )
        )
        if already_configured and not force:
            return

        self.input_dims = 3
        self.image_in_channels = image_in_channels
        if not self.uses_volume_adapter:
            self.volume_adapter = None
            return

        if self.volume_adapter_type != "spatial_z_attention":
            raise ValueError(f"Unsupported volume_adapter_type: {self.volume_adapter_type}")
        self.volume_adapter = TrainableVolumeToSamAdapter(
            in_channels=image_in_channels,
            input_z_size=self.input_z_size,
            adapter_channels=self.volume_adapter_channels,
            res_blocks=self.volume_adapter_res_blocks,
            norm=self.volume_adapter_norm)
        self.volume_adapter.requires_grad_(self.train_volume_adapter)

    def ensure_input_adapter(self, image_in_channels: int) -> None:
        """Keep the existing adapter when compatible, otherwise build one for the requested input."""
        self.configure_input_adapter(image_in_channels=image_in_channels, force=False)

    def prepare_sam_image(self, x_in: torch.Tensor) -> torch.Tensor:
        """Map BCZYX input to the 1-channel SAM image used by the ViT backbone."""
        if x_in.dim() != 5:
            raise ValueError(f"SamUNetR expects 5D BCZYX input, got shape {tuple(x_in.shape)}.")
        if not self.uses_volume_adapter:
            if x_in.shape[2] != 1:
                raise ValueError(
                    f"2D degenerate mode expects Z=1 after dataset resampling, got Z={x_in.shape[2]}."
                )
            x_sam_1ch = x_in.squeeze(2)
            if x_sam_1ch.shape[1] != 1:
                raise ValueError(
                    f"2D degenerate mode expects one selected biological channel, got C={x_sam_1ch.shape[1]}."
                )
            return x_sam_1ch
        if self.volume_adapter is None:
            raise RuntimeError("volume_adapter is required when input_z_size > 1.")
        return self.volume_adapter(x_in)

    def extract_embed_feat(
        self,
        x_in: torch.Tensor,
        use_lora: bool = False,
    ) -> torch.Tensor:
        """Run ViT only and return pre-neck embedding (embed_dim channels); LoRA off by default."""
        prev = None
        if self.lora_modules:
            prev = self.lora_modules[0].enabled
            for m in self.lora_modules:
                m.enabled = use_lora
        try:
            x_sam_1ch = self.prepare_sam_image(x_in)
            x_vit = torch.repeat_interleave(x_sam_1ch, 3, dim=1)
            vit_out = self.vit(x_vit, get_hidden_state=[])
            return vit_out[0] if isinstance(vit_out, tuple) else vit_out
        finally:
            if prev is not None:
                for m in self.lora_modules:
                    m.enabled = prev

    def forward(
        self,
        x_in,
        blur_skip: bool = False,
        use_lora: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward SamUNetR.

        ``blur_skip`` defaults to False (training uses the raw encoder1 skip). Evaluation
        stitch paths pass True when ``use_image_skip`` is enabled. Ignored when skip is
        disabled. ``use_lora=False`` forces frozen-ViT features for cell encoding.
        """
        prev = None
        if use_lora is not None and self.lora_modules:
            prev = self.lora_modules[0].enabled
            for m in self.lora_modules:
                m.enabled = use_lora
        try:
            x_sam_1ch = self.prepare_sam_image(x_in)
            x_vit = torch.repeat_interleave(x_sam_1ch, 3, dim=1)
            x, hidden_states_out = self.vit(x_vit, get_hidden_state=self.hidden_layer_hooks)

            # decoder2: UnetrUpBlock(+encoder1 skip) or UnetrUpBlockNoSkip, per use_image_skip.
            x_skip = None
            if self.use_image_skip:
                if blur_skip:
                    decoder_skip_sigma = 3.0
                    decoder_skip_kernel = 2 * int(3 * decoder_skip_sigma) + 1
                    original_hw = x_sam_1ch.shape[-2:]
                    x_sam_1ch_small = F.interpolate(
                        x_sam_1ch, size=(256, 256), mode='bilinear', align_corners=False)
                    x_skip = gaussian_blur(
                        x_sam_1ch_small, kernel_size=decoder_skip_kernel, sigma=decoder_skip_sigma)
                    x_skip = F.interpolate(
                        x_skip, size=original_hw, mode='bilinear', align_corners=False)
                else:
                    x_skip = x_sam_1ch
            hv_mask_feat = self.upsampler.hv_mask_decoder(hidden_states_out, x_skip=x_skip)
            cyto_feat = self.upsampler.cyto_type_decoder(hidden_states_out, x_skip=x_skip)
            raw_mask = self.upsampler.nuc_mask_predictor(hv_mask_feat) # C=1
            raw_hv = self.upsampler.nuc_hv_predictor(hv_mask_feat) # C=2
            cyto_type = self.upsampler.cyto_type_predictor(cyto_feat) # C=num_class

            return {'refined_mask': raw_mask, # Keep this for further refinement
                    'refined_hv': raw_hv, # Keep this for further refinement
                    'raw_mask': raw_mask, 
                    'raw_hv': raw_hv, 
                    'cyto_type': cyto_type, 
                    'embed_feat': x, 
                    }
        finally:
            if prev is not None:
                for m in self.lora_modules:
                    m.enabled = prev

class PredictionHead(nn.Module):
    """
    A dedicated convolutional prediction head after the shared decoder.
    """
    def __init__(self,
                 spatial_dims: int,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 block_num: int = 3,
                 norm_name: tuple | str = "instance",
                 res_block: bool = True):
        """
        Build a thicker output branch without duplicating the full UNETR upsampler.
        """
        super().__init__()

        blocks = []
        next_in_channels = in_channels
        for _ in range(block_num):
            blocks.append(UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=next_in_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            ))
            next_in_channels = hidden_channels

        self.feature_extractor = nn.Sequential(*blocks)
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=next_in_channels,
            out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert shared decoder features into per-pixel cyto type logits.
        """
        x = self.feature_extractor(x)
        return self.out(x)


class UnetrUpBlockNoSkip(nn.Module):
    """
    Final UNETR upsample stage without concatenating an image-level skip feature.

    Keeps a ``transp_conv`` attribute name compatible with old UnetrUpBlock checkpoints;
    ``conv_block`` is reinitialized when loading old skip-concat weights (shape mismatch).
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        upsample_kernel_size: int,
        norm_name: tuple | str,
        res_block: bool = True,
    ) -> None:
        super().__init__()
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_kernel_size,
            conv_only=True,
            is_transposed=True,
        )
        self.conv_block = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample decoder features without an image skip branch."""
        return self.conv_block(self.transp_conv(x))

class UNETR_UpSampler(nn.Module):
    """
    UNETR decoder over ViT multi-scale tokens.

    ``use_image_skip=True``: encoder1 + UnetrUpBlock decoder2 (image skip, older edition).
    ``use_image_skip=False``: UnetrUpBlockNoSkip decoder2 (ViT tokens only).
    """
    def __init__(self,
                 out_channels: int|None, # None returns decoder features for shared heads
                 hidden_size: int = 1024, # equal to the encoder_embed_dim in SAM
                 feature_size: int = 16,
                 norm_name: tuple | str = "instance",
                 conv_block: bool = True,
                 res_block: bool = True,
                 spatial_dims: int = 2,
                 use_image_skip: bool = True,
                 in_channels: int = 1,
                 **kwargs):
        super().__init__()
        self.use_image_skip = use_image_skip

        # Image-level skip stem used only when decoder2 concatenates encoder1 features.
        if use_image_skip:
            self.encoder1 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=feature_size,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )
        else:
            self.encoder1 = None

        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        # decoder2: with-skip (UnetrUpBlock) or no-skip (UnetrUpBlockNoSkip).
        if use_image_skip:
            self.decoder2 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size * 2,
                out_channels=feature_size,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
        else:
            self.decoder2 = UnetrUpBlockNoSkip(
                spatial_dims=spatial_dims,
                in_channels=feature_size * 2,
                out_channels=feature_size,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
        self.out = None if out_channels is None else UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=out_channels)
        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims)) # Move the hidden-size dimension to the second axis.

    def proj_feat(self, x):
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(
        self,
        hidden_states_out: torch.Tensor,
        x_skip: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode ViT multi-scale tokens; optionally fuse encoder1 image skip into decoder2."""
        x2 = hidden_states_out[0]
        enc2 = self.encoder2(self.proj_feat(x2))
        x3 = hidden_states_out[1]
        enc3 = self.encoder3(self.proj_feat(x3))
        x4 = hidden_states_out[2]
        enc4 = self.encoder4(self.proj_feat(x4))
        x5 = hidden_states_out[3]
        dec4 = self.proj_feat(x5)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        if self.use_image_skip:
            out = self.decoder2(dec1, self.encoder1(x_skip))
        else:
            out = self.decoder2(dec1)
        if self.out is None:
            return out
        final_res = self.out(out)
        return final_res


class LoRAQKV(nn.Module):
    """Wrap SAM Attention.qkv with trainable low-rank updates on Q and V only.

    Mirrors SAMed ``_LoRA_qkv``: frozen ``W_qkv(x)`` plus ``B_q A_q(x)`` / ``B_v A_v(x)``.
    SAM tokens are shaped ``(B, H, W, C)``, so updates use ellipsis indexing.
    Toggle ``enabled`` to bypass the residual (cell encoding uses the frozen backbone).
    """

    def __init__(
        self,
        qkv: nn.Linear,
        linear_a_q: nn.Linear,
        linear_b_q: nn.Linear,
        linear_a_v: nn.Linear,
        linear_b_v: nn.Linear,
    ) -> None:
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return qkv projection; when enabled, add LoRA residuals to Q and V slices."""
        qkv = self.qkv(x)
        if not self.enabled:
            return qkv
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv = qkv.clone()
        qkv[..., : self.dim] = qkv[..., : self.dim] + new_q
        qkv[..., -self.dim :] = qkv[..., -self.dim :] + new_v
        return qkv

    def adapter_parameters(self):
        """A/B weights for Q and V (excludes frozen base ``qkv``)."""
        yield from self.linear_a_q.parameters()
        yield from self.linear_b_q.parameters()
        yield from self.linear_a_v.parameters()
        yield from self.linear_b_v.parameters()


def inject_lora_into_vit(
    vit: nn.Module,
    rank: int = 4,
    lora_layers: Sequence[int] | None = None,
) -> list[LoRAQKV]:
    """Freeze ViT weights and replace ``blocks[i].attn.qkv`` with ``LoRAQKV`` (SAMed surgery).

    New A/B layers keep ``requires_grad=True``; call sites that later freeze the whole ViT
    must re-enable A/B grads explicitly.
    """
    if rank <= 0:
        raise ValueError(f"LoRA rank must be > 0, got {rank}.")
    if not hasattr(vit, "blocks"):
        raise TypeError("vit must expose .blocks like SAM ImageEncoderViT.")

    n_blocks = len(vit.blocks)
    if lora_layers is None:
        target_layers = list(range(n_blocks))
    else:
        target_layers = sorted({
            (idx if idx >= 0 else n_blocks + idx) for idx in lora_layers
        })
        for resolved in target_layers:
            if not (0 <= resolved < n_blocks):
                raise ValueError(f"Invalid LoRA layer index for depth {n_blocks}: {resolved}.")

    for param in vit.parameters():
        param.requires_grad = False

    injected: list[LoRAQKV] = []
    for layer_i, blk in enumerate(vit.blocks):
        if layer_i not in target_layers:
            continue
        w_qkv = blk.attn.qkv
        if isinstance(w_qkv, LoRAQKV):
            raise RuntimeError(f"LoRA already injected at block {layer_i}.")
        dim = w_qkv.in_features
        linear_a_q = nn.Linear(dim, rank, bias=False)
        linear_b_q = nn.Linear(rank, dim, bias=False)
        linear_a_v = nn.Linear(dim, rank, bias=False)
        linear_b_v = nn.Linear(rank, dim, bias=False)
        nn.init.kaiming_uniform_(linear_a_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(linear_a_v.weight, a=math.sqrt(5))
        nn.init.zeros_(linear_b_q.weight)
        nn.init.zeros_(linear_b_v.weight)
        lora_qkv = LoRAQKV(w_qkv, linear_a_q, linear_b_q, linear_a_v, linear_b_v)
        blk.attn.qkv = lora_qkv
        injected.append(lora_qkv)

    vit.lora_rank = rank
    vit.lora_layers = target_layers
    return injected


def lora_state_dict(lora_modules: Sequence[LoRAQKV]) -> dict[str, torch.Tensor]:
    """Pack LoRA A/B weights into a SAMed-compatible flat dict."""
    w_as: list[nn.Linear] = []
    w_bs: list[nn.Linear] = []
    for module in lora_modules:
        w_as.extend((module.linear_a_q, module.linear_a_v))
        w_bs.extend((module.linear_b_q, module.linear_b_v))
    state: dict[str, torch.Tensor] = {}
    for i, linear in enumerate(w_as):
        state[f"w_a_{i:03d}"] = linear.weight.detach().cpu()
    for i, linear in enumerate(w_bs):
        state[f"w_b_{i:03d}"] = linear.weight.detach().cpu()
    return state


def load_lora_state_dict(
    lora_modules: Sequence[LoRAQKV],
    state: dict[str, torch.Tensor],
) -> None:
    """Load weights produced by ``lora_state_dict`` (SAMed A/B key layout)."""
    w_as: list[nn.Linear] = []
    w_bs: list[nn.Linear] = []
    for module in lora_modules:
        w_as.extend((module.linear_a_q, module.linear_a_v))
        w_bs.extend((module.linear_b_q, module.linear_b_v))
    for i, linear in enumerate(w_as):
        linear.weight.data.copy_(
            state[f"w_a_{i:03d}"].to(dtype=linear.weight.dtype, device=linear.weight.device))
    for i, linear in enumerate(w_bs):
        linear.weight.data.copy_(
            state[f"w_b_{i:03d}"].to(dtype=linear.weight.dtype, device=linear.weight.device))
