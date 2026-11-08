from scripts_formal.utils.elements_samunetr import SamUNetR
from scripts_formal.utils.elements_data import GridImageDataset
from scripts_formal.utils.elements_vit import retrieve_sam_encoder
from scripts_formal.utils.elements_postprocess import SamProcessor
from scripts_formal.utils.elements_loss import DeepLosses, LossTracker

import os
import shutil
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from typing import Callable, Optional, Literal, Sequence
from tqdm import tqdm
from datetime import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
import optuna

CHECKPOINT_COMPONENT_PREFIXES: dict[str, tuple[str, ...]] = {
    'vit': ('vit.',),
    'upsampler': ('upsampler.hv_mask_decoder.', 'upsampler.cyto_type_decoder.'),
    'volume_adapter': ('volume_adapter.',),
    'predictors': (
        'upsampler.nuc_mask_predictor.',
        'upsampler.nuc_hv_predictor.',
        'upsampler.cyto_type_predictor.',
    ),
}

# Keys dropped when loading checkpoints from older / full-SAM layouts.
# encoder1 is not listed: when use_image_skip=False those keys are absent from
# model_state and dropped by shape/key mismatch; when True they load normally.
OBSOLETE_CHECKPOINT_SUBSTRINGS: tuple[str, ...] = (
    'vit.neck.',
)

class LUMATrainer:
    """
    A class to train and use a SamUNetR model for semantic and instance segmentation.
    """

    def __init__(self, backbone_type="vit_h",
                 checkpoint=None, 
                 dataset_path=None, 
                 device=None,
                 num_class:int=1,
                 use_prediction_head: bool = False,
                 prediction_head_channels: Optional[int] = None,
                 prediction_head_blocks: int = 3,
                 input_z_size: int = 10,
                 volume_adapter_type: str = "spatial_z_attention",
                 volume_adapter_channels: int = 8,
                 volume_adapter_res_blocks: int = 4,
                 volume_adapter_norm: str = "groupnorm",
                 train_volume_adapter: bool = True,
                 vit_adapt: str = "none",
                 lora_rank: int = 4,
                 lora_layers: Optional[Sequence[int]] = None,
                 use_image_skip: bool = True,
                 sam_skip_channels: int = 1,
                 init_model: bool = True,
                 **kwargs):
        """
        Initializes the SamClarifier.

        Args:
            backbone_type (str): The type of ViT model to use (e.g., "vit_h", "vit_l", "vit_b").
            checkpoint (str): Path to the SAM encoder checkpoint.
            dataset_path (str, optional): Path to the GridImageDataset.
            device (str, optional): The device to run the model on ('cuda' or 'cpu').
            num_class (int): The number of instance classes for segmentation.
            use_prediction_head (bool): Whether to use thicker output heads for the hv/mask branch in SamUNetR.
            prediction_head_channels (int, optional): Hidden channels for thicker hv/mask output heads.
            prediction_head_blocks (int): Number of convolutional blocks in thicker hv/mask output heads.
            input_z_size (int): Fixed z depth used for 3D input volumes.
            volume_adapter_type (str): 3D-to-2D adapter type used before SAM ViT.
            volume_adapter_channels (int): Hidden channels inside the volume adapter.
            volume_adapter_res_blocks (int): Number of residual 3D blocks in the volume adapter.
            volume_adapter_norm (str): Normalization type inside the volume adapter.
            train_volume_adapter (bool): Whether the volume adapter remains trainable.
            vit_adapt (str): ViT adaptation: ``none`` (frozen) or ``lora`` (SAMed-style Q/V LoRA).
            lora_rank (int): LoRA rank when ``vit_adapt='lora'``.
            lora_layers (Sequence[int], optional): Block indices for LoRA; ``None`` means all blocks.
            use_image_skip (bool): Whether UNETR decoder2 uses encoder1 image skip (UnetrUpBlock).
            sam_skip_channels (int): Channels of the image skip into encoder1 when use_image_skip=True.
            **kwargs: Accepted for backward compatibility (e.g. obsolete ``classifier_params``).
        """
        
        # Parameters for the SAM model
        self.build_parameters = {
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
        
        self.backbone_type = backbone_type
        self.checkpoint = checkpoint
        self.model = None
        assert num_class >= 1, "num_class must be at least 1 to account for the baseline class."
        self.num_instance_class = num_class
        self.use_prediction_head = use_prediction_head
        self.prediction_head_channels = prediction_head_channels
        self.prediction_head_blocks = prediction_head_blocks
        self.input_dims = 3
        self.input_z_size = input_z_size
        self.image_channel_count = 0
        self.model_input_channel_count = 1
        self.volume_adapter_type = volume_adapter_type
        self.volume_adapter_channels = volume_adapter_channels
        self.volume_adapter_res_blocks = volume_adapter_res_blocks
        self.volume_adapter_norm = volume_adapter_norm
        self.train_volume_adapter = train_volume_adapter
        self.vit_adapt = vit_adapt
        self.lora_rank = lora_rank
        self.lora_layers = list(lora_layers) if lora_layers is not None else None
        self.use_image_skip = use_image_skip
        self.sam_skip_channels = sam_skip_channels
        
        if device:
            try:
                self.device = torch.device(device)
            except:
                print(f"Invalid device specified: {device}. Falling back to automatic device detection.")
                self.device = None
        if not device:
            # --- LOGIC FOR ROBUST DEVICE DETECTION ---
            if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES'] and torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        
        dataset_configured = self.config_data(dataset_path=dataset_path)
        if not dataset_configured:
            self.dataset_path = None
        else:
            self.dataset_path = dataset_path
            if init_model:
                self.config_model()

        self.config_postprocessor()

    def train(self,
              title:str='training_session',
              description:str='',
              log_dir:str='./training_logs',
              epochs=100, num_workers=0, batch_size=4,
              accumulation_steps:int=1,
              learning_rate=1e-4, train_val_ratio=0.8,
              optimizer_type:str='adam',
              target_image_channel=[2],
              model_save_name="best_model.pth",
              train_encoder:bool|list=False, 
              # A list selects block indexes; True trains all blocks; False disables encoder training.
              encoder_learning_rate:float=1e-5,
              use_amp:bool=True,
              use_bf16:bool=False,
              scheduler_T_max:int|None=None,
              load_model_path:Optional[str]=None,
              load_typemap_upsampler:bool=True,
              load_components:Optional[Sequence[str]|Literal['all']]=None,
              patience:int=25,
              uncertainty_weight:bool=True,
              loss_type_hv:str='cpdynamics_hv',
              hv_loss_weight:float=2.0,
              class_weight:list[float]|None=None,
              **kwargs
              ) -> dict:
        """
        Trains the SamUNetR model.

        Args:
            epochs (int): Number of training epochs.
            num_workers (int): Number of workers for the DataLoader.
            batch_size (int): Training batch size.
            learning_rate (float): Learning rate for the main model optimizers.
            train_val_ratio (float): Ratio to split the dataset into training and validation sets.
            model_save_path (str): Path to save the best model.
            hv_loss_weight (float): Optimization weight applied to raw_hv loss in total loss.
            class_weight (list[float], optional): Fixed cyto type class weights used instead of batch weights.

        Returns:
            dict: A history of training and validation losses.
        """
        # --- 1. Setup session log directory ---
        if kwargs.get('model_save_path', None):
            model_save_name = kwargs['model_save_path']
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        session_log_dir = os.path.join(log_dir, title + '_' + current_time)
        os.makedirs(session_log_dir, exist_ok=True)
        model_save_path = os.path.join(session_log_dir, model_save_name)
        log_txt_path = os.path.join(session_log_dir, "log.txt")
        loss_png_path = os.path.join(session_log_dir, "loss.png")
        # ------------- 2. Definition of Helper Functions-------------
        def logger(text):
            """Prints to console and appends to log file."""
            with open(log_txt_path, "a", encoding="utf-8") as f:
                f.write(str(text) + "\n")
        
        def plot_loss():
            """Plots validation loss history."""
            plt.switch_backend('Agg') # Use non-interactive backend
            fig, ax = plt.subplots(figsize=(10, 6))
            
            history = self.loss_tracker.history
            metrics = self.loss_tracker.loss_to_report
            
            # Generate colors
            cmap = plt.get_cmap('tab10')
            colors = {m: cmap(i) for i, m in enumerate(metrics)}
            
            for metric in metrics:
                # Val Loss
                if metric in history['val_loss'] and len(history['val_loss'][metric]) > 0:
                    val_vals = np.array(history['val_loss'][metric])
                    val_vals = np.where(val_vals > 1e-10, val_vals, 1e-10)
                    val_log = np.log10(val_vals)
                    # Handle case where val might be shorter than train during current epoch
                    current_epochs = range(1, len(val_log) + 1)
                    ax.plot(current_epochs, val_log, label=f'{metric} (Val)', 
                            color=colors[metric], linestyle='--', linewidth=0.8, alpha=0.8)

            ax.set_xlabel('Epochs')
            ax.set_ylabel('Log10 Loss')
            ax.set_title('Validation Loss (Log Scale)')
            ax.set_xlim(1, epochs)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            
            plt.tight_layout()
            plt.savefig(loss_png_path, dpi=150)
            plt.close(fig)

        def get_trainable_model_params(exclude_vit:bool=True):
            """Collect trainable model parameters while optionally keeping encoder params separate."""
            vit_param_ids = {id(p) for p in self.model.vit.parameters()} if exclude_vit else set()
            return [p for p in self.model.parameters() if p.requires_grad and id(p) not in vit_param_ids]
        
        def get_loss_params(loss_modules:dict[str, DeepLosses]):
            """Collect trainable loss-module parameters for uncertainty weighting."""
            loss_params = []
            for loss_module in loss_modules.values():
                loss_params.extend(list(loss_module.parameters()))
            return loss_params

        # ---------- 3. Log Session Info / optionally stage non-/gpfs/ data ----------
        local_dataset_cache = None
        original_dataset_path = self.dataset_path
        try:
            job_id = os.environ.get("SLURM_JOB_ID", "local_test")
            logger(f"--- Training Session: {title} ---")
            logger(f"Job ID: {job_id}")
            logger(f"Description: {description}")

            # If the dataset realpath is outside /gpfs/, copy it into this session for fast IO.
            if self.dataset_path is not None and self._dataset_requires_local_staging(self.dataset_path):
                local_dataset_cache = self._stage_dataset_to_session_cache(
                    source_path=self.dataset_path,
                    session_log_dir=session_log_dir,
                    logger=logger,
                )
                self.config_data(dataset_path=local_dataset_cache)
                self.dataset_path = local_dataset_cache

            # ---------- 4. Configure DataLoaders ----------
            if self.dataset is None:
                logger("Dataset not prepared. Please run config_data() first.")
                return
            target_image_channel = self.configure_model_input(target_image_channel)
            dataset_size = len(self.dataset)
            train_size = int(train_val_ratio * dataset_size)
            val_size = dataset_size - train_size
            if val_size < 1:
                logger("Validation set size is too small. Consider adjusting train_val_ratio or dataset size.")
                return
        
            train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available()) if val_size > 0 else None

            # ------- 5. Configure Model Parameters -------
            if load_model_path:
                logger(f"Loading SamClarifier model from {load_model_path}...")
                self.load_model(
                    load_model_path,
                    load_typemap_upsampler=load_typemap_upsampler,
                    load_components=load_components,
                )
            self.model.to(self.device)
            self.model.requires_grad_(True)

            # Freeze all ViT parameters first; vit_adapt='lora' unfreezes only the LoRA branches.
            self.model.vit.requires_grad_(False)
            train_lora = bool(self.model.lora_modules)
            if train_lora:
                if isinstance(train_encoder, bool) and train_encoder:
                    logger(
                        "vit_adapt='lora' keeps the SAM ViT backbone frozen; "
                        "ignoring train_encoder=True and training LoRA only."
                    )
                elif isinstance(train_encoder, list) and len(train_encoder) > 0:
                    logger(
                        "vit_adapt='lora' keeps the SAM ViT backbone frozen; "
                        f"ignoring train_encoder={train_encoder} and training LoRA only."
                    )
                encoder_params = []
                for m in self.model.lora_modules:
                    m.enabled = True
                    for p in m.adapter_parameters():
                        p.requires_grad = True
                        encoder_params.append(p)
                n_lora = sum(p.numel() for p in encoder_params)
                logger(
                    f"Training SAMed-style LoRA (rank={self.lora_rank}, "
                    f"trainable params={n_lora}) with learning rate: {encoder_learning_rate}"
                )
                train_encoder = False
            elif isinstance(train_encoder, bool):
                if train_encoder:
                    self.model.vit.requires_grad_(True)
                    logger(f"Training fully ViT Encoder with learning rate: {encoder_learning_rate}")
                else:
                    logger("ViT Encoder is fully frozen.")
            elif isinstance(train_encoder, list):
                # Deduplicate the list and validate the indexes.
                unique_blocks = list(set(train_encoder))
                block_indexes = [i for i in unique_blocks if isinstance(i, int)]
                for ind in range(len(block_indexes)):
                    if block_indexes[ind] < 0:
                        block_indexes[ind] = len(self.model.vit.blocks) + block_indexes[ind]  # Support negative indexes.
                max_idx = len(self.model.vit.blocks) - 1
                valid_blocks = [idx for idx in block_indexes if 0 <= idx <= max_idx]
            
                if len(valid_blocks) != len(unique_blocks):
                    logger(f"Warning: Found invalid indices in train_encoder list. Expected range 0-{max_idx}. Filtered valid ones: {valid_blocks}")
            
                if len(valid_blocks) > 0:
                    # Unfreeze only the blocks at the selected indexes.
                    for idx in valid_blocks:
                        self.model.vit.blocks[idx].requires_grad_(True)
                    logger(f"Training partial ViT Encoder (blocks: {valid_blocks}) with learning rate: {encoder_learning_rate}")
                else:
                    train_encoder = False  # Disable encoder training if no valid blocks remain.
                    logger("No valid ViT blocks to train, ViT Encoder is fully frozen.")
        
            # ----- 6. AMP Scaler Initialization -----
            if self.device.type != 'cuda':
                use_amp = False 
                logger("Device is not CUDA, disabling AMP.")
            scaler = torch.amp.GradScaler(enabled=(use_amp and not use_bf16))
            if use_amp:
                logger("Automatic Mixed Precision (AMP) is ENABLED.")
            else:
                logger("Automatic Mixed Precision (AMP) is DISABLED.")

            # ----- 7. Configure Losses, Optimizers, Schedulers -----
            self.config_loss_tracker()

            if loss_type_hv.endswith('_logits'):
                loss_type_hv = loss_type_hv.replace('_logits', '')
                logger(f"Converted legacy HV loss name to signed regression loss: {loss_type_hv}")
            logger(f"HV loss weight in total objective: {hv_loss_weight}")

            criterion = {
                'raw_mask': DeepLosses(
                    loss_type='dicebce_logits',
                    dicebce_weights=None if uncertainty_weight else [0.5, 0.5],
                ),
                'raw_hv': DeepLosses(loss_type=f'{loss_type_hv}', hv_weight=hv_loss_weight), 
                'cyto_type': DeepLosses(loss_type='ce_dice_loss_logits', cedice_weights=[0.5, 0.5] if uncertainty_weight else [0.5, 0.5]),
                # Temporarily disable cyto_type uncertainty weighting to test suitable loss ratios.
            }

            for key in criterion.keys():
                criterion[key].to(self.device)

            if class_weight is None:
                fixed_cyto_type_class_weights = None
                logger("Using cyto type class weights from each data batch.")
            else:
                fixed_cyto_type_class_weights = torch.tensor(class_weight, dtype=torch.float32, device=self.device)
                logger(f"Using fixed cyto type class weights: {fixed_cyto_type_class_weights.detach().cpu().tolist()}")

            main_params = get_trainable_model_params(exclude_vit=True) + get_loss_params(criterion)
            if optimizer_type.lower() == 'adam':
                optimizer = {
                    'model': optim.AdamW([{'params': main_params, 'lr': learning_rate}]),
                }
            elif optimizer_type.lower() == 'sgd':
                optimizer = {
                    'model': optim.SGD([{'params': main_params, 'lr': learning_rate}],
                                       momentum=0.9, weight_decay=1e-4),
                }
        
            if train_lora:
                if len(encoder_params) > 0:
                    if optimizer_type.lower() == 'adam':
                        optimizer['encoder'] = optim.AdamW(encoder_params, lr=encoder_learning_rate)
                    else:
                        optimizer['encoder'] = optim.SGD(encoder_params, lr=encoder_learning_rate, momentum=0.9, weight_decay=1e-4)
            elif (isinstance(train_encoder, bool) and train_encoder) or (isinstance(train_encoder, list) and len(train_encoder) > 0):
                encoder_params = list(filter(lambda p: p.requires_grad, self.model.vit.parameters()))
                if len(encoder_params) > 0:
                    if optimizer_type.lower() == 'adam':
                        optimizer['encoder'] = optim.AdamW(encoder_params, lr=encoder_learning_rate)
                    else:
                        optimizer['encoder'] = optim.SGD(encoder_params, lr=encoder_learning_rate, momentum=0.9, weight_decay=1e-4)

            t_max_real = scheduler_T_max if scheduler_T_max is not None else epochs
            scheduler = {k: optim.lr_scheduler.CosineAnnealingLR(v, T_max=t_max_real) for k, v in optimizer.items()}

            def optimizer_zero_grad_all():
                for opt in optimizer.values():
                    opt.zero_grad()
        
            # ------------- Start Training Loop -------------
            logger(f"Starting training on {self.device}...")

            patience_counter = 0

            for epoch in range(epochs):
                strt = time.time()
                self.model.train() 
                base_desc = f"Epoch {epoch+1}/{epochs} [Train]"
                train_pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                                  desc=base_desc)
                for batch_idx, batch in train_pbar:
                    if self.device.type == 'cuda':
                        torch.cuda.reset_peak_memory_stats(self.device)

                    batch:dict[str, torch.Tensor]
                    img, mask_gt, hv_gt, cyto_type_gt = \
                    self.prepare_model_input(batch, target_image_channel), \
                        batch['mask'][:, 1:, :, :].to(self.device), \
                        batch['hv_map'].to(self.device), \
                        batch['type_cyto_map'].to(self.device)

                    mask_gt_ = mask_gt.clone().bool()
                    if fixed_cyto_type_class_weights is None:
                        cyto_type_class_weights = batch.get('cyto_type_class_weights')
                        if cyto_type_class_weights is not None:
                            cyto_type_class_weights = cyto_type_class_weights.to(self.device)
                    else:
                        cyto_type_class_weights = fixed_cyto_type_class_weights

                    with torch.amp.autocast(device_type='cuda', enabled=use_amp, dtype=torch.bfloat16 if use_bf16 else torch.float16):
                        res_dict:dict[str, torch.Tensor] = self.model.forward(img)

                        raw_mask_pred, raw_hv_pred, cyto_type_pred = \
                        [res_dict[k].to(self.device) for k in [
                            'raw_mask', 'raw_hv', 'cyto_type']]
                    
                        
                    
                        loss_raw_mask = criterion['raw_mask'](raw_mask_pred, mask_gt)
                        self.loss_tracker.tensor_loss['raw_mask'] = loss_raw_mask.item()
                        loss_raw_hv = criterion['raw_hv'](raw_hv_pred, hv_gt, mask_gt_)
                        self.loss_tracker.tensor_loss['raw_hv'] = loss_raw_hv.item()
                        loss_cyto_type = criterion['cyto_type'](
                            cyto_type_pred,
                            cyto_type_gt,
                            mask_gt_,
                            class_weights=cyto_type_class_weights)
                        self.loss_tracker.tensor_loss['cyto_type'] = loss_cyto_type.item()
                        total_loss = loss_raw_mask + loss_raw_hv + loss_cyto_type
                        
                        total_loss = total_loss / accumulation_steps
                
                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        logger(f"WARNING: Batch {batch_idx} in Epoch {epoch+1} produced NaN/Inf loss! Skipping step.")
                        # Optionally save problematic batch data for offline analysis.
                        # torch.save(batch, f"debug_nan_batch_epoch{epoch}_{batch_idx}.pt")
                        optimizer_zero_grad_all()
                        continue

                    # --- Backpropagation with Gradient Scaling ---
                    # Scales loss. calls backward() on scaled loss to create scaled gradients.
                    scaler.scale(total_loss).backward()

                    if self.device.type == 'cuda':
                        # max_memory_allocated is showing the peak memory usage so far
                        peak_mem_usage_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
                        train_pbar.set_description(f"{base_desc} Peak Mem: {peak_mem_usage_gb:.2f}GB")

                    if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1 == len(train_loader)):
                        opt_2_step = list(optimizer.keys())

                        # Gradient Clipping (Don't forget to unscale first)
                        for opt_name in opt_2_step:
                            scaler.unscale_(optimizer[opt_name])
                        '''torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)'''
                        parameters_to_clip = []
                        for opt_name in opt_2_step:
                            for group in optimizer[opt_name].param_groups:
                                parameters_to_clip.extend(group['params'])
                    
                        torch.nn.utils.clip_grad_norm_(parameters_to_clip, max_norm=1.0)

                        for opt_name in opt_2_step:
                            scaler.step(optimizer[opt_name])
                        scaler.update()
                        for opt_name in opt_2_step:
                            optimizer[opt_name].zero_grad()

                    run_loss = self.loss_tracker.update_running('train_loss', img.size(0))
                    train_pbar.set_postfix(run_loss)

                if optimizer_type.lower() == 'sgd': # Only sgd needs annealing
                    for key in scheduler.keys():
                        scheduler[key].step()

                log_text, _ = self.loss_tracker.log_epoch('train_loss')
                logger(f'Epoch {epoch+1}/{epochs} - Training Loss: {log_text}')

                if val_loader:
                    self.model.eval()
                    val_pbar = tqdm(enumerate(val_loader), total=len(val_loader),
                                    desc=f"Epoch {epoch+1}/{epochs} [Val]")
                    with torch.no_grad():
                        for batch_idx, batch in val_pbar:
                            img, mask_gt, hv_gt, cyto_type_gt = \
                                self.prepare_model_input(batch, target_image_channel), \
                                    batch['mask'][:, 1:, :, :].to(self.device), \
                                    batch['hv_map'].to(self.device), \
                                    batch['type_cyto_map'].to(self.device)

                            mask_gt_ = mask_gt.clone().bool()
                            if fixed_cyto_type_class_weights is None:
                                cyto_type_class_weights = batch.get('cyto_type_class_weights')
                                if cyto_type_class_weights is not None:
                                    cyto_type_class_weights = cyto_type_class_weights.to(self.device)
                            else:
                                cyto_type_class_weights = fixed_cyto_type_class_weights
                        
                            with torch.amp.autocast(device_type='cuda', enabled=use_amp, dtype=torch.bfloat16 if use_bf16 else torch.float16):
                                res_dict:dict[str, torch.Tensor] = self.model.forward(img)
                            
                                raw_mask_pred, raw_hv_pred, cyto_type_pred = \
                                [res_dict[k].to(self.device) for k in [
                                    'raw_mask', 'raw_hv', 'cyto_type']]
                           
                                loss_raw_mask = criterion['raw_mask'](raw_mask_pred, mask_gt, eval_mode=True)
                                self.loss_tracker.tensor_loss['raw_mask'] = loss_raw_mask.item()
                                loss_raw_hv = criterion['raw_hv'](raw_hv_pred, hv_gt, mask_gt_, eval_mode=True)
                                self.loss_tracker.tensor_loss['raw_hv'] = loss_raw_hv.item()
                                loss_cyto_type = criterion['cyto_type'](
                                    cyto_type_pred,
                                    cyto_type_gt,
                                    mask_gt_,
                                    class_weights=cyto_type_class_weights,
                                    eval_mode=True)
                                self.loss_tracker.tensor_loss['cyto_type'] = loss_cyto_type.item()
                                total_loss = loss_raw_mask + loss_raw_hv + loss_cyto_type

                            run_loss = self.loss_tracker.update_running('val_loss', img.size(0))
                            val_pbar.set_postfix(run_loss)
                
                    log_text, best_epoch = self.loss_tracker.log_epoch('val_loss')   
                    logger(f"Epoch {epoch+1}/{epochs} - Validation Loss: {log_text}")
                
                    # Save the best model based on validation loss
                    if best_epoch['total'] and not kwargs.get('optuna_trial'):
                        save_dir = os.path.dirname(model_save_path)
                        if save_dir and not os.path.exists(save_dir):
                            os.makedirs(save_dir, exist_ok=True)
                        torch.save(self.model.state_dict(), model_save_path)
                        self.save_checkpoint_bundle(os.path.dirname(model_save_path))
                        logger(f"Best model saved to {model_save_path}")
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logger(f"No improvement for {patience} consecutive epochs. Early stopping triggered.")
                            break

                else: # No validation loader
                    _, _ = self.loss_tracker.log_epoch('val_loss')
                    save_dir = os.path.dirname(model_save_path)
                    if save_dir and not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    torch.save(self.model.state_dict(), model_save_path)
                    self.save_checkpoint_bundle(os.path.dirname(model_save_path))
                    logger(f"Model saved to {model_save_path} (Epoch {epoch+1})")

                try:
                    plot_loss()
                except Exception as e:
                    logger(f"Warning: Failed to update loss plot. Error: {e}")

                end = time.time()
                epoch_duration = end - strt
                logger(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds.\n")

                if kwargs.get('optuna_trial'):
                    trial = kwargs['optuna_trial']
                    current_val_loss = self.loss_tracker.history['val_loss']['total'][-1]
                    trial.report(current_val_loss, epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()            

            logger(f"Training finished. Final model may be saved if no validation loader is provided.")
            if not val_loader:
                final_model_save_path = model_save_path.replace('.pth', '_final.pth')
                torch.save(self.model.state_dict(), final_model_save_path)
                self.save_checkpoint_bundle(session_log_dir, filename_prefix='final')
            best_model_path = model_save_path if val_loader else final_model_save_path
            return self.loss_tracker.history, best_model_path
        finally:
            # Drop the session-local dataset copy after training or early exit.
            if local_dataset_cache is not None:
                self._cleanup_session_dataset_cache(local_dataset_cache, logger)
                self.dataset_path = original_dataset_path

    def exchange_dataset(self, dataset_path:str):
        """
        Exchanges the current dataset with a new one.

        Args:
            dataset_path (str): Path to the new GridImageDataset.

        Returns:
            bool: True if dataset is successfully exchanged, False otherwise.
        """
        dataset_configured = self.config_data(dataset_path=dataset_path)
        if dataset_configured:
            print(f"Dataset exchanged successfully. Will reconfigure model and loss tracker!")
            self.config_model()
        else:
            print("Failed to exchange dataset. Please provide a valid dataset path.")
        return

    def load_model(
        self,
        model_path,
        load_typemap_upsampler: bool = True,
        load_components: Optional[Sequence[str] | Literal['all']] = None,
    ):
        """
        Loads a model state dictionary from a file.

        Adapts older checkpoints that still contain ViT neck weights, and reinits tensors
        whose shapes changed across skip / no-skip decoder2 layouts.

        Args:
            model_path (str): Path to the model file.
            load_typemap_upsampler (bool): Whether to load cyto type decoder and predictor weights.
            load_components (sequence or 'all'): Optional subset of components to load
                ('vit', 'upsampler', 'volume_adapter', 'predictors'). Defaults to full checkpoint.
        Returns:
            None
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} does not exist.")
        state_dict: dict[str, torch.Tensor] = torch.load(model_path, map_location='cpu')
        self._configure_input_adapter_from_state_dict(state_dict)

        if load_components is not None and load_components != 'all':
            state_dict = self._filter_state_dict_by_components(state_dict, load_components)
            if not load_typemap_upsampler:
                state_dict = self._filter_cyto_type_keys(state_dict)
            state_dict, dropped_keys = self._adapt_state_dict_to_current_model(
                state_dict, self.model.state_dict())
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            if unexpected_keys:
                raise RuntimeError(f"Unexpected keys when loading components {load_components}: {unexpected_keys}")
            if dropped_keys:
                print(f"Dropped obsolete/mismatched checkpoint keys ({len(dropped_keys)}): "
                      f"{dropped_keys[:8]}{'...' if len(dropped_keys) > 8 else ''}")
            if missing_keys:
                print(f"Loaded components {list(load_components)}; uninitialized keys remain: {missing_keys}")
            self.model.to(self.device)
            print(f"Model loaded from {model_path} (components={list(load_components)}).")
            return

        if not load_typemap_upsampler:
            state_dict = self._filter_cyto_type_keys(state_dict)
        adapted_state_dict, dropped_keys = self._adapt_state_dict_to_current_model(
            state_dict, self.model.state_dict())
        missing_keys, unexpected_keys = self.model.load_state_dict(adapted_state_dict, strict=False)
        if unexpected_keys:
            raise RuntimeError(f"Unexpected keys after checkpoint adaptation: {unexpected_keys}")
        if dropped_keys:
            print(f"Dropped obsolete/mismatched checkpoint keys ({len(dropped_keys)}): "
                  f"{dropped_keys[:8]}{'...' if len(dropped_keys) > 8 else ''}")
        if not load_typemap_upsampler:
            print("Loaded model while reinitializing cyto type decoder and predictor.")
        elif missing_keys:
            print(f"Loaded model with reinitialized keys: {missing_keys}")
        self.model.to(self.device)
        print(f"Model loaded from {model_path}.")
        return

    def save_checkpoint_bundle(self, save_dir: str, filename_prefix: str = '') -> None:
        """Save full, backbone, and adapter checkpoints from the current model."""
        if self.model is None:
            raise RuntimeError("Cannot save checkpoints before the model is configured.")
        os.makedirs(save_dir, exist_ok=True)
        prefix = f"{filename_prefix}_" if filename_prefix else ''
        state_dict = self.model.state_dict()
        buckets = self._split_state_dict_by_components(state_dict)

        backbone_keys = (
            CHECKPOINT_COMPONENT_PREFIXES['vit']
            + CHECKPOINT_COMPONENT_PREFIXES['upsampler']
            + CHECKPOINT_COMPONENT_PREFIXES['predictors']
        )
        backbone_state = {
            key: value for key, value in state_dict.items()
            if any(key.startswith(prefix) for prefix in backbone_keys)
        }
        backbone_path = os.path.join(save_dir, f'{prefix}backbone_2d.pth')
        torch.save(backbone_state, backbone_path)

        adapter_path = os.path.join(save_dir, f'{prefix}adapter_3d.pth')
        if buckets['volume_adapter']:
            torch.save(buckets['volume_adapter'], adapter_path)
        return

    def config_data(self, dataset_path=None):
        if dataset_path and os.path.exists(dataset_path):
            try:
                self.datagrid = GridImageDataset.load(dataset_path)
                self.dataset = self.datagrid.merged_dataset['patch']
                self.sc_dataset = self.datagrid.merged_dataset['sc']
                self.input_dims = 3
                self.image_channel_count = getattr(self.dataset, 'image_channel_count', 0) if self.dataset is not None else 0
            except Exception as e:
                print(f"Error loading dataset from {dataset_path}: {e}")
                self.dataset = None
                self.sc_dataset = None
                return False
            return True
        else:
            print("No dataset path provided. Please provide a valid dataset path.")
            self.dataset = None
            self.sc_dataset = None
            return False

    def config_model(self,
                     use_prediction_head: Optional[bool] = None,
                     prediction_head_channels: Optional[int] = None,
                     prediction_head_blocks: Optional[int] = None):
        """Configures the SamUNetR model and its output-head style."""
        self.img_encoder = retrieve_sam_encoder(model_type=self.backbone_type, checkpoint=self.checkpoint)
        use_prediction_head = self.use_prediction_head if use_prediction_head is None else use_prediction_head
        prediction_head_channels = self.prediction_head_channels if prediction_head_channels is None else prediction_head_channels
        prediction_head_blocks = self.prediction_head_blocks if prediction_head_blocks is None else prediction_head_blocks
        
        self.model = SamUNetR(
            vit_encoder=self.img_encoder,
            hidden_size=self.build_parameters[self.backbone_type]["encoder_embed_dim"],
            hidden_layer_hooks=[i+1 for i in self.build_parameters[self.backbone_type]["encoder_global_attn_indexes"]],
            feature_size=16,
            spatial_dims=2,
            num_class=self.num_instance_class,
            use_prediction_head=use_prediction_head,
            prediction_head_channels=prediction_head_channels,
            prediction_head_blocks=prediction_head_blocks,
            input_z_size=self.input_z_size,
            image_in_channels=self.model_input_channel_count,
            volume_adapter_type=self.volume_adapter_type,
            volume_adapter_channels=self.volume_adapter_channels,
            volume_adapter_res_blocks=self.volume_adapter_res_blocks,
            volume_adapter_norm=self.volume_adapter_norm,
            train_volume_adapter=self.train_volume_adapter,
            vit_adapt=self.vit_adapt,
            lora_rank=self.lora_rank,
            lora_layers=self.lora_layers,
            use_image_skip=self.use_image_skip,
            sam_skip_channels=self.sam_skip_channels)
        
        self.model.cpu()
        mode_label = '3D (volume adapter)' if self.input_z_size > 1 else '2D degenerate'
        skip_label = 'with image skip' if self.use_image_skip else 'no image skip'
        adapt_label = f', vit_adapt={self.vit_adapt}'
        if self.vit_adapt == 'lora':
            adapt_label += f'(r={self.lora_rank})'
        print(f"--- SamClarifier initialized on device: {self.device} "
              f"(Zt={self.input_z_size}, {mode_label}, {skip_label}{adapt_label}) ---")

    def config_postprocessor(self, **kwargs) -> None:
        """
        Configures the post-processor with specified parameters.

        Args:
            **kwargs: Keyword arguments for SamProcessor configuration.
                * min_mask_size (int): Minimum size of the mask to keep.
                * min_major_axis_length (int): Minimum major axis length for instance filtering.
                * conduct_size_filter (bool): Whether to conduct size filtering based on min_major_axis_length.

        Returns:
            None
        """
        min_mask_size = kwargs.get('min_mask_size', 3000)
        min_major_axis_length = kwargs.get('min_major_axis_length', 'auto')
        conduct_size_filter = kwargs.get('conduct_size_filter', True)
        conduct_mask_filter = kwargs.get('conduct_mask_filter', True)

        self.postprocessor = SamProcessor(
            min_mask_size=min_mask_size,
            min_major_axis_length=min_major_axis_length,
            conduct_size_filter=conduct_size_filter,
            conduct_mask_filter=conduct_mask_filter
        )

    def config_loss_tracker(self, report_cyto_type: bool | None = None):
        if report_cyto_type is None:
            report_cyto_type = self.num_instance_class > 1
        loss_to_report = ['raw_mask', 'raw_hv']
        if report_cyto_type:
            loss_to_report.append('cyto_type')
        self.loss_tracker = LossTracker(
            loss_items=['raw_mask', 'raw_hv', 'cyto_type'],
            loss_to_report=loss_to_report,
            tracked_items=[],
            exclude_items=[]
        )

    def configure_model_input(self, target_image_channel: list[int]|int) -> list[int]:
        """Configure dataset z-resampling and SamUNetR input adapters for selected biological channels."""
        if isinstance(target_image_channel, int):
            target_image_channel = [target_image_channel]
        target_image_channel = list(target_image_channel)
        self.model_input_channel_count = len(target_image_channel)
        zt = self.model.input_z_size if self.model is not None else self.input_z_size
        self.input_z_size = zt

        if hasattr(self.dataset, 'configure_3d_input'):
            self.dataset.configure_3d_input(target_image_channel, input_z_size=zt)
        if hasattr(self.sc_dataset, 'configure_3d_input'):
            self.sc_dataset.configure_3d_input(target_image_channel, input_z_size=zt)
        if self.model is not None and hasattr(self.model, 'configure_input_adapter'):
            self.model.input_z_size = zt
            self.model.configure_input_adapter(
                image_in_channels=self.model_input_channel_count,
                force=True,
            )
        return target_image_channel

    def prepare_model_input(self, batch: dict[str, torch.Tensor], target_image_channel: list[int]) -> torch.Tensor:
        """Select biological channels from BCZYX batches and move them to the training device."""
        image = batch['image']
        if image.dim() == 5:
            image_channel_indices = batch.get('image_channel_indices')
            if image_channel_indices is not None:
                first_indices = image_channel_indices[0].detach().cpu().tolist()
                if first_indices == target_image_channel:
                    return image.to(self.device)
            return image[:, target_image_channel, :, :, :].to(self.device)
        raise ValueError(f"Unsupported batch image shape; expected BCZYX, got {tuple(image.shape)}")

    def _configure_input_adapter_from_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Materialize input adapters before loading checkpoints that contain adapter weights."""
        if self.model is None or not hasattr(self.model, 'configure_input_adapter'):
            return
        input_channels = self.model_input_channel_count
        if 'volume_adapter.channel_adapter.weight' in state_dict:
            input_channels = state_dict['volume_adapter.channel_adapter.weight'].shape[1]
        self.model_input_channel_count = input_channels
        self.input_dims = 3
        self.model.configure_input_adapter(image_in_channels=input_channels, force=True)

    def _stage_dataset_to_session_cache(
        self,
        source_path: str,
        session_log_dir: str,
        logger: Callable[[str], None],
    ) -> str:
        """
        Copy a non-/gpfs/ dataset into ``session_log_dir/local_dataset_cache`` for local IO.

        Reloading should be done by the caller via ``config_data`` on the returned path.
        """
        source_real = os.path.realpath(source_path)
        cache_dir = os.path.join(session_log_dir, 'local_dataset_cache')
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)

        # Log copy plan before the potentially long transfer.
        logger(
            f"Dataset real path is outside /gpfs/: {source_real}. "
            f"Staging a temporary copy to {cache_dir} ..."
        )
        print(
            f"Dataset real path is outside /gpfs/: {source_real}. "
            f"Staging a temporary copy to {cache_dir} ..."
        )
        copy_start = time.time()
        shutil.copytree(source_real, cache_dir)
        copy_seconds = time.time() - copy_start
        logger(f"Dataset staging finished in {copy_seconds:.1f}s -> {cache_dir}")
        print(f"Dataset staging finished in {copy_seconds:.1f}s -> {cache_dir}")
        return cache_dir

    def _cleanup_session_dataset_cache(
        self,
        cache_dir: str,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        """Delete a session-local dataset cache created by ``_stage_dataset_to_session_cache``."""
        if not cache_dir or not os.path.exists(cache_dir):
            return
        message = f"Removing temporary local dataset cache: {cache_dir}"
        if logger is not None:
            logger(message)
        print(message)
        shutil.rmtree(cache_dir)

    @staticmethod
    def _dataset_requires_local_staging(dataset_path: str) -> bool:
        """Return True when the resolved dataset path is not under ``/gpfs/``."""
        real_path = os.path.realpath(dataset_path)
        return not (real_path == '/gpfs' or real_path.startswith('/gpfs/'))

    @staticmethod
    def _filter_cyto_type_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Drop cyto-type decoder/predictor weights (e.g. when class count differs across datasets)."""
        skip_keywords = ('cyto_type_decoder', 'cyto_type_predictor')
        return {
            key: value for key, value in state_dict.items()
            if not any(skip_keyword in key for skip_keyword in skip_keywords)
        }

    @staticmethod
    def _adapt_state_dict_to_current_model(
        state_dict: dict[str, torch.Tensor],
        model_state: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        """
        Adapt an older / full-SAM checkpoint to the current SamUNetR layout.

        Drops obsolete modules (ViT neck) and tensors whose shapes no longer match
        (e.g. decoder2.conv_block across skip / no-skip layouts). Keys absent from the
        current model (including encoder1 when use_image_skip=False) are dropped.
        Compatible tensors such as decoder2.transp_conv are kept.
        """
        adapted: dict[str, torch.Tensor] = {}
        dropped: list[str] = []
        for key, value in state_dict.items():
            if any(substring in key for substring in OBSOLETE_CHECKPOINT_SUBSTRINGS):
                dropped.append(key)
                continue
            if key not in model_state:
                dropped.append(key)
                continue
            if tuple(model_state[key].shape) != tuple(value.shape):
                dropped.append(key)
                continue
            adapted[key] = value
        return adapted, dropped

    @staticmethod
    def _filter_state_dict_by_components(
        state_dict: dict[str, torch.Tensor],
        components: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        """Keep only keys belonging to the requested checkpoint components."""
        prefixes: list[str] = []
        for component in components:
            if component not in CHECKPOINT_COMPONENT_PREFIXES:
                raise ValueError(
                    f"Unknown checkpoint component '{component}'. "
                    f"Expected one of: {sorted(CHECKPOINT_COMPONENT_PREFIXES)} or 'all'."
                )
            prefixes.extend(CHECKPOINT_COMPONENT_PREFIXES[component])
        return {
            key: value for key, value in state_dict.items()
            if any(key.startswith(prefix) for prefix in prefixes)
        }

    @staticmethod
    def _split_state_dict_by_components(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Split a full state dict into named component buckets."""
        buckets = {name: {} for name in CHECKPOINT_COMPONENT_PREFIXES}
        for key, value in state_dict.items():
            for component, prefixes in CHECKPOINT_COMPONENT_PREFIXES.items():
                if any(key.startswith(prefix) for prefix in prefixes):
                    buckets[component][key] = value
                    break
        return buckets
