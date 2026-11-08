import torch
from functools import partial
from copy import deepcopy
from segment_anything.modeling import Sam, PromptEncoder, MaskDecoder, TwoWayTransformer, ImageEncoderViT

class ModifiedViTEncoder(ImageEncoderViT):
    """
    SAM ImageEncoderViT without the final neck.

    Forward returns raw ViT patch tokens as (B, embed_dim, H, W), e.g. 1024-d for vit_l.
    ``strip_neck`` drops the inherited neck module after SAM checkpoint loading so old
    ``neck.*`` weights can be loaded first and then discarded.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed_dim: int = int(kwargs.get('embed_dim', 768))
        # Expose pre-neck channel count for downstream consumers.
        self.out_channels: int = self.embed_dim

    def strip_neck(self) -> None:
        """Remove the SAM neck module after checkpoint weights have been applied."""
        if hasattr(self, 'neck'):
            delattr(self, 'neck')

    def forward(self, x: torch.Tensor, get_hidden_state: list=[]) -> torch.Tensor:
        """
        Forward pass through the modified ViT encoder. Will only return hidden states when a layer index is specified in `get_hidden_state`.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            get_hidden_state (list): List of indices at which to retrieve hidden states.
        Returns:
            torch.Tensor: Output tensor after processing through the encoder (B, embed_dim, H, W).
        """
        try:
            get_hidden_state = list(get_hidden_state) # Ensure it's a list
        except TypeError:
            get_hidden_state = []

        x = self.patch_embed(x) # B C H W -> B H W C
        if self.pos_embed is not None:
            x = x + self.pos_embed

        hidden_states_out = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if idx + 1 in get_hidden_state:
                # Store the hidden state at the specified index
                hidden_states_out.append(x)

        # Pre-neck ViT features: B H W C -> B C H W (C = embed_dim, e.g. 1024).
        x = x.permute(0, 3, 1, 2).contiguous()

        if hidden_states_out:
            return x, hidden_states_out
        else:
            return x

def retrieve_sam_encoder(model_type="vit_h", checkpoint="sam_vit_h_4b8939.pth") -> torch.nn.Module:
    """
    Retrieve a SAM encoder based on the specified checkpoint and model type.

    Loads the official SAM checkpoint (including neck weights for shape compatibility),
    then strips the neck so the returned encoder exposes raw ViT embeddings only.

    Args:
        checkpoint (str): Path to the SAM model checkpoint.
        model_type (str): Type of the SAM model (e.g., 'vit_h', 'vit_l', 'vit_b').

    Returns:
        SamPredictor: An instance of the SAM predictor with the specified model.
    """

    build_parameters = {
        "vit_h": {
            "encoder_embed_dim": 1280,
            "encoder_depth": 32,
            "encoder_num_heads": 16,
            "encoder_global_attn_indexes": [7, 15, 23, 31],
        },
        "vit_l": {
            "encoder_embed_dim": 1024,
            "encoder_depth": 24,
            "encoder_num_heads": 16,
            "encoder_global_attn_indexes": [5, 11, 17, 23],
        },
        "vit_b": {
            "encoder_embed_dim": 768,
            "encoder_depth": 12,
            "encoder_num_heads": 12,
            "encoder_global_attn_indexes": [2, 5, 8, 11],
        },
    }

    def build_sam(checkpoint=None, model_type="vit_h"):
        """
        Build a SAM model based on the specified model type and checkpoint.

        Args:
            checkpoint (str): Path to the SAM model checkpoint.
            model_type (str): Type of the SAM model (e.g., 'vit_h', 'vit_l', 'vit_b').

        Returns:
            Sam: An instance of the SAM model.
        """
        if model_type not in build_parameters:
            raise ValueError(f"Model type '{model_type}' is not recognized.")
        
        params = build_parameters[model_type]
        return _build_sam(
            encoder_embed_dim=params["encoder_embed_dim"],
            encoder_depth=params["encoder_depth"],
            encoder_num_heads=params["encoder_num_heads"],
            encoder_global_attn_indexes=params["encoder_global_attn_indexes"],
            checkpoint=checkpoint,
        )

    def _build_sam(
        encoder_embed_dim,
        encoder_depth,
        encoder_num_heads,
        encoder_global_attn_indexes,
        checkpoint=None,
        ):
        prompt_embed_dim = 256
        image_size = 1024
        vit_patch_size = 16
        image_embedding_size = image_size // vit_patch_size
        sam = Sam(
            image_encoder=ModifiedViTEncoder(
                depth=encoder_depth,
                embed_dim=encoder_embed_dim,
                img_size=image_size,
                mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=encoder_num_heads,
                patch_size=vit_patch_size,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=encoder_global_attn_indexes,
                window_size=14,
                out_chans=prompt_embed_dim,
            ),
            prompt_encoder=PromptEncoder(
                embed_dim=prompt_embed_dim,
                image_embedding_size=(image_embedding_size, image_embedding_size),
                input_image_size=(image_size, image_size),
                mask_in_chans=16,
            ),
            mask_decoder=MaskDecoder(
                num_multimask_outputs=3,
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )
        sam.eval()
        if checkpoint is not None:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f)
            # Keep neck in the module graph only while loading official SAM weights.
            sam.load_state_dict(state_dict)
        return sam
    
    
    sam = build_sam(checkpoint=checkpoint, model_type=model_type)
    image_encoder = deepcopy(sam.image_encoder)
    image_encoder.strip_neck()
    image_encoder.eval()
    return image_encoder
