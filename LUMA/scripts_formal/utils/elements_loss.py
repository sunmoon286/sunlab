import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepLosses(nn.Module):
    def __init__(self, loss_type='dice', 
                 dicebce_weights:list[float]|None=[0.5, 0.5],
                 cedice_weights:list[float]|None=[0.5, 0.5],
                 label_smoothing:float=0.0,
                 ignore_non_instance:bool=True,
                 dice_smooth:float|None=None,
                 hv_cell_weight:float=5.0,
                 hv_background_weight:float=1.0,
                 hv_weight:float=1.0):
        # Possible loss types: 'dice', 'dicebce', 'focal'
        # Dice ratio is the weight of dice loss in dicebce
        super(DeepLosses, self).__init__()
        self.loss_type = loss_type
        self.alpha = 0.8
        self.gamma = 2
        self.epsilon = 1e-6
        self.smooth = 1
        self.dicebce_weights = dicebce_weights
        self.cedice_weights = cedice_weights
        self.label_smoothing = label_smoothing
        self.ignore_non_instance = ignore_non_instance
        self.dice_smooth = self.smooth if dice_smooth is None else dice_smooth
        self.hv_cell_weight = hv_cell_weight
        self.hv_background_weight = hv_background_weight
        self.hv_weight = hv_weight

        self.log_vars_dicebce = nn.Parameter(torch.zeros(2))
        self.log_vars_cedice = nn.Parameter(torch.zeros(2))
        
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                    dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                    dtype=torch.float32).view(1, 1, 3, 3)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

    def dice_loss(self, y_pred:torch.Tensor, y_true:torch.Tensor) -> torch.Tensor:
        intersection = (y_pred * y_true).sum()
        DCE = 1. - (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return DCE
    
    def bce_loss(self, y_pred:torch.Tensor, y_true:torch.Tensor) -> torch.Tensor:
        BCE = F.binary_cross_entropy(y_pred, y_true, reduction='mean')
        return BCE
    
    def focal_loss(self, y_pred:torch.Tensor, y_true:torch.Tensor):
        BCE = F.binary_cross_entropy(y_pred, y_true, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        FCL = self.alpha * (1 - BCE_EXP) ** self.gamma * BCE
        return FCL
    
    def l2_loss(self, y_pred:torch.Tensor, y_true:torch.Tensor):
        return F.mse_loss(y_pred, y_true, reduction='mean')
    
    def weighted_l2_loss(self, y_pred:torch.Tensor, y_true:torch.Tensor, weight:torch.Tensor|None=None):
        squared_error = (y_pred - y_true).pow(2)
        if weight is None:
            return squared_error.mean()
        return (squared_error * weight).sum() / weight.sum().clamp_min(self.epsilon)
    
    def hv_weight_map(self, target_tensor:torch.Tensor, mask:torch.Tensor|None=None):
        if mask is None:
            return None
        if mask.dim() == target_tensor.dim() - 1:
            mask = mask.unsqueeze(1)
        mask = mask.to(device=target_tensor.device, dtype=target_tensor.dtype)
        if mask.shape[1] == 1 and target_tensor.shape[1] != 1:
            mask = mask.expand(-1, target_tensor.shape[1], -1, -1)
        return torch.where(
            mask > 0,
            torch.full_like(target_tensor, self.hv_cell_weight),
            torch.full_like(target_tensor, self.hv_background_weight))
    
    def cross_entropy_loss(self, y_pred:torch.Tensor, y_true:torch.Tensor):
        return F.cross_entropy(y_pred, y_true, reduction='mean')

    def hv_ensemble_loss(self, y_pred:torch.Tensor, y_true:torch.Tensor, mask:torch.Tensor=None):
        hv_weight = self.hv_weight_map(y_pred, mask)
        loss_main = self.weighted_l2_loss(y_pred, y_true, hv_weight)
        if not self.loss_type.startswith('hovernet_hv'):
            return loss_main * self.hv_weight
        grad_pred = self.get_gradient(y_pred)
        grad_true = self.get_gradient(y_true)
        grad_weight = self.hv_weight_map(grad_pred, mask)
        loss_grad = self.weighted_l2_loss(grad_pred, grad_true, grad_weight)
        return (loss_main + loss_grad) * self.hv_weight
    
    def dicebce_loss(self, y_pred:torch.Tensor, y_true:torch.Tensor, eval_mode:bool=False):
        DCE = self.dice_loss(y_pred, y_true)
        BCE = self.bce_loss(y_pred, y_true)
        if not eval_mode and self.dicebce_weights is None:
            precision_dice = torch.exp(-self.log_vars_dicebce[0])
            loss_dice = precision_dice * DCE + self.log_vars_dicebce[0]
            precision_bce = torch.exp(-self.log_vars_dicebce[1])
            loss_bce = precision_bce * BCE + self.log_vars_dicebce[1]
            return loss_dice + loss_bce
        else:
            weights = self.dicebce_weights if self.dicebce_weights is not None else [0.5, 0.5]
            return weights[0] * DCE + weights[1] * BCE
        
    def dicebce_loss_logits(
        self,
        y_pred:torch.Tensor,
        y_true:torch.Tensor,
        mask:torch.Tensor|None=None,
        eval_mode:bool=False,
    ):
        if mask is not None:
            mask = mask.bool()
            if mask.dim() == y_pred.dim() - 1:
                mask = mask.unsqueeze(1)
            pixel_mask = mask.expand_as(y_pred)
            y_pred_m = y_pred[pixel_mask]
            y_true_m = y_true[pixel_mask]
        else:
            y_pred_m = y_pred.contiguous().view(-1)
            y_true_m = y_true.contiguous().view(-1)

        BCE = F.binary_cross_entropy_with_logits(y_pred_m, y_true_m, reduction='mean')
        y_pred_prob = torch.sigmoid(y_pred_m)
        DCE = self.dice_loss(y_pred_prob, y_true_m)

        if not eval_mode and self.dicebce_weights is None:
            precision_dice = torch.exp(-self.log_vars_dicebce[0])
            loss_dice = precision_dice * DCE + self.log_vars_dicebce[0]
            precision_bce = torch.exp(-self.log_vars_dicebce[1])
            loss_bce = precision_bce * BCE + self.log_vars_dicebce[1]
            loss = loss_dice + loss_bce
        else:
            weights = self.dicebce_weights if self.dicebce_weights is not None else [0.5, 0.5]
            loss = weights[0] * DCE + weights[1] * BCE
        return loss
    
    def ce_dice_loss_logits(self,
                            y_pred:torch.Tensor,
                            y_true:torch.Tensor,
                            mask:torch.Tensor|None=None,
                            class_weights:torch.Tensor|None=None,
                            eval_mode:bool=False) -> torch.Tensor:
        """Calculate masked softmax-CE plus class-weighted macro soft-Dice for cyto type logits."""
        batch_size, num_classes, height, width = y_pred.shape
        if mask is not None:
            if mask.dim() == 4:
                mask = mask.squeeze(1)
            instance_mask = mask.bool()
        elif self.ignore_non_instance:
            instance_mask = y_true.sum(dim=1) > 0
        else:
            instance_mask = torch.ones(batch_size, height, width, dtype=torch.bool, device=y_pred.device)

        target_idx = y_true.argmax(dim=1).long()
        target_idx_ce = target_idx.masked_fill(~instance_mask, -100)
        if bool(instance_mask.any().item()):
            ce = F.cross_entropy(
                y_pred,
                target_idx_ce,
                label_smoothing=self.label_smoothing,
                ignore_index=-100,
                reduction='mean')
            if torch.isnan(ce):
                ce = y_pred.new_zeros(())
        else:
            ce = y_pred.new_zeros(())

        probs = F.softmax(y_pred, dim=1)
        valid_mask = instance_mask.unsqueeze(1).float()
        probs_m = probs * valid_mask
        target_m = y_true.float() * valid_mask
        class_weights = None if class_weights is None else class_weights.to(device=y_pred.device, dtype=y_pred.dtype)

        dice_terms = []
        dice_weights = []
        for class_idx in range(num_classes):
            pred_class = probs_m[:, class_idx]
            true_class = target_m[:, class_idx]
            present_per_sample = true_class.flatten(1).sum(dim=1) > 0
            if not bool(present_per_sample.any().item()):
                continue
            intersection = (pred_class * true_class).sum()
            denominator = pred_class.sum() + true_class.sum()
            dice_terms.append(
                1.0 - (2.0 * intersection + self.dice_smooth) / (denominator + self.dice_smooth)
            )
            if class_weights is None:
                dice_weights.append(y_pred.new_ones(()))
            elif class_weights.dim() == 1:
                dice_weights.append(class_weights[class_idx])
            else:
                dice_weights.append(class_weights[present_per_sample, class_idx].mean())

        if len(dice_terms) == 0:
            dce = y_pred.new_zeros(())
        else:
            dice_terms = torch.stack(dice_terms)
            dice_weights = torch.stack(dice_weights)
            dice_weights = dice_weights / dice_weights.mean().clamp_min(self.epsilon)
            dce = (dice_terms * dice_weights).sum() / dice_weights.sum().clamp_min(self.epsilon)

        if not eval_mode and self.cedice_weights is None:
            precision_dice = torch.exp(-self.log_vars_cedice[0])
            loss_dice = precision_dice * dce + self.log_vars_cedice[0]
            precision_ce = torch.exp(-self.log_vars_cedice[1])
            loss_ce = precision_ce * ce + self.log_vars_cedice[1]
            return loss_dice + loss_ce
        else:
            weights = self.cedice_weights if self.cedice_weights is not None else [0.5, 0.5]
            return weights[0] * dce + weights[1] * ce
        
    def supcon_loss(self, embeddings:torch.Tensor, labels:torch.Tensor, temperature:float=0.07):
        """
        Supervised Contrastive Loss implementation.
        Args:
            embeddings (torch.Tensor): The projection features, shape: [N, feat_dim].
            labels (torch.Tensor): The ground truth labels, shape: [N].
            temperature (float): The temperature hyperparameter.
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device

        # --- Unsupervised (SimCLR) Case ---
        if labels is None:
            # Assuming the batch is structured as [view1_samples, view2_samples].
            assert batch_size % 2 == 0, "For unsupervised SupCon, batch size must be even (pairs of views)."
            half_batch_size = batch_size // 2
            # Creates labels like [0, 1, 2, ..., N-1, 0, 1, 2, ..., N-1]
            labels = torch.arange(half_batch_size).repeat(2)
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Create the similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T)
        
        # Create the label mask: mask_{i,j} is 1 if labels[i] == labels[j]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Mask out the diagonal (self-similarity)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size, device=device).view(-1, 1),
            0
        )
        mask = mask * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(similarity_matrix / temperature) * logits_mask
        log_prob = similarity_matrix / temperature - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # Compute mean of log-likelihood over positive pairs
        # mask.sum(1) is the number of positive pairs for each anchor
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        
        # The loss is the negative of this mean log-likelihood
        loss = -mean_log_prob_pos.mean()
        
        return loss
    ### END MODIFIED
    
    def get_gradient(self, x:torch.Tensor):
        grad_h, grad_v = x[:, 0:1, :, :], x[:, 1:2, :, :]
        grad_h_x = F.conv2d(grad_h, self.sobel_x.to(x.device), padding=1)
        grad_h_y = F.conv2d(grad_h, self.sobel_y.to(x.device), padding=1)
        grad_v_x = F.conv2d(grad_v, self.sobel_x.to(x.device), padding=1)
        grad_v_y = F.conv2d(grad_v, self.sobel_y.to(x.device), padding=1)
        grad = torch.cat([grad_h_x, grad_h_y, grad_v_x, grad_v_y], dim=1)
        return grad

    def forward(self, y_pred:torch.Tensor, y_true:torch.Tensor, mask:torch.Tensor=None, 
                eval_mode:bool=False, **kwargs) -> torch.Tensor:
        cedice_loss_types = ['ce_dice_loss_logits', 'ce_dice_logits', 'cedice_logits']
    
        if self.loss_type not in ['cross_entropy', 'ce', 'supcon']:
            assert y_pred.shape == y_true.shape, "Prediction and ground truth must have the same shape."
        
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            y_pred = y_pred.float()
            # Keep y_true as Long for cross entropy.
            if self.loss_type in ['cross_entropy', 'ce']:
                y_true = y_true.long()
            else:
                y_true = y_true.float()

            # [Added] Detect and replace NaN/Inf to prevent CUDA assertion failures.
            if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
                # Strategy: replace NaN with 0.5 (or another safe value) to avoid a crash.
                # Alternatively, return a huge loss to invalidate this iteration.
                # Here, use safe replacements so a large loss signals poor predictions to the optimizer.
                y_pred = torch.nan_to_num(y_pred, nan=0.5, posinf=1.0, neginf=0.0)

            # [Changed] Clamp only for non-logits losses, since logits are unbounded.
            if self.loss_type in ['bce', 'dicebce', 'focal'] and 'logits' not in self.loss_type:
                y_pred = torch.clamp(y_pred, min=1e-7, max=1.0-1e-7)

            if self.loss_type in cedice_loss_types:
                return self.ce_dice_loss_logits(
                    y_pred,
                    y_true,
                    mask=mask,
                    class_weights=kwargs.get('class_weights'),
                    eval_mode=eval_mode)

            if mask is not None:
                mask = mask.bool()
                if mask.dim() == y_pred.dim() - 1:
                    mask = mask.unsqueeze(1)
                y_pred_masked = y_pred[mask.expand_as(y_pred)]
                y_true_masked = y_true[mask.expand_as(y_true)]
            
            else:
                y_pred_masked = y_pred.contiguous().view(-1)
                y_true_masked = y_true.contiguous().view(-1)

            if self.loss_type in [
                'hovernet_hv', 'cpdynamics_hv',
                'hovernet_hv_logits', 'cpdynamics_hv_logits',
            ]:
                return self.hv_ensemble_loss(y_pred, y_true, mask)

            elif self.loss_type == 'dicebce_logits': # Dice Loss + BCE Loss (Logits)
                return self.dicebce_loss_logits(y_pred, y_true, mask=mask, eval_mode=eval_mode)
            
            elif self.loss_type == 'dice': # Dice Loss
                return self.dice_loss(y_pred_masked, y_true_masked)
            
            elif self.loss_type == 'bce': # BCE Loss
                return self.bce_loss(y_pred_masked, y_true_masked)
            
            elif self.loss_type == 'dicebce': # Dice Loss + BCE Loss
                return self.dicebce_loss(y_pred_masked, y_true_masked, eval_mode=eval_mode)
            
            elif self.loss_type == 'focal': # Focal Loss
                return self.focal_loss(y_pred_masked, y_true_masked)
            
            elif self.loss_type in ['l2', 'l2_regress']: # L2 Loss
                return self.l2_loss(y_pred_masked, y_true_masked)
            
            elif self.loss_type in ['cross_entropy', 'ce']: # Cross Entropy Loss
                return self.cross_entropy_loss(y_pred, y_true)
            
            ### MODIFIED: Added 'supcon' loss type handling
            elif self.loss_type == 'supcon': # Supervised Contrastive Loss
                return self.supcon_loss(
                    embeddings=y_pred, 
                    labels=y_true, 
                    temperature=kwargs.get('temperature', 0.07)
                )
            ### END MODIFIED
            
            else:
                return 0
            
class LossTracker:
    def __init__(self, loss_items:list=['raw_mask', 'raw_hv', 'cyto_type'],
                 loss_to_report:list=['raw_mask', 'raw_hv', 'cyto_type'],
                 tracked_items:list=[],
                 exclude_items:list=None):

        self.best_val_loss = float('inf')
        self.loss_items = loss_items
        self.extra_tracked_items = tracked_items
        self.exclude_items = [] if exclude_items is None else exclude_items

        # --- changed: stable, order-preserving de-dup instead of set() ---
        combined = (loss_to_report or []) + (self.extra_tracked_items or [])
        loss_to_report = list(dict.fromkeys(combined))  # preserves order
        # ---------------------------------------------------------------

        self.loss_to_report = loss_to_report if loss_to_report else loss_items
        self.best_val_loss = {
            'total': float('inf'),
        }
        self.best_val_loss.update({item: float('inf') for item in self.extra_tracked_items})

        self.history = {'train_loss': {'total': []}, 'val_loss': {'total': []}}
        for item in self.loss_to_report:
            self.history['train_loss'][item] = []
            self.history['val_loss'][item] = []
        
        self.running = {'train_loss': {'total': 0.0, 'count': 0}, 'val_loss': {'total': 0.0, 'count': 0}}
        for item in self.loss_items:
            self.running['train_loss'][item] = 0.0
            self.running['val_loss'][item] = 0.0

        self.tensor_loss:dict[str, torch.Tensor] = {}

    def update_running(self, phase:str, batch_num:int):
        assert phase in ['train_loss', 'val_loss'], "Phase must be 'train_loss' or 'val_loss'."
        for item in self.loss_items:
            if item in self.tensor_loss.keys():
                self.running[phase][item] += self.tensor_loss[item] * batch_num
                if item not in self.exclude_items:
                    self.running[phase]['total'] += self.tensor_loss[item] * batch_num
        self.running[phase]['count'] += batch_num
        return {k:v for (k, v) in self.tensor_loss.items()}
    
    def log_epoch(self, phase:str) -> tuple[str, dict]:

        best_epoch = {item: False for item in self.best_val_loss.keys()}

        assert phase in ['train_loss', 'val_loss'], "Phase must be 'train_loss' or 'val_loss'."
        if self.running[phase]['count'] == 0:
            for item in self.loss_items + ['total']:
                if item in self.history[phase].keys():
                    self.history[phase][item].append(0.0)
                self.running[phase][item] = 0.0
            return "No data to log.", best_epoch
        for item in self.loss_items + ['total']:
            if item in self.history[phase].keys():
                self.history[phase][item].append(self.running[phase][item] / self.running[phase]['count'])
            self.running[phase][item] = 0.0
        self.running[phase]['count'] = 0

        if phase == 'val_loss':
            for metric in ['total'] + self.extra_tracked_items:
                if self.history['val_loss'][metric][-1] < self.best_val_loss[metric]:
                    self.best_val_loss[metric] = self.history['val_loss'][metric][-1]
                    best_epoch[metric] = True
        
        epoch_log = {item: self.history[phase][item][-1] for item in ['total'] + self.loss_to_report}
        printed_log = ', '.join([f"{k}: {v:.4f}" for k, v in epoch_log.items()])

        return printed_log, best_epoch

