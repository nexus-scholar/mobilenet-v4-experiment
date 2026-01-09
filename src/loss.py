import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositeLoss(nn.Module):
    """
    Composite loss combining classification, distillation, and segmentation losses.

    Components:
    - L_ce: Cross-entropy loss for classification
    - L_dist: Cosine Embedding Loss for knowledge distillation (feature alignment)
    - L_seg: BCE loss for segmentation masks
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, temperature: float = 1.0, ignore_index: int = -1):
        """
        Args:
            alpha: Weight for distillation loss (default: 1.0)
            beta: Weight for segmentation loss (default: 1.0)
            temperature: Unused for Cosine Loss (kept for config compatibility).
            ignore_index: Label index to ignore in CE loss (default: -1)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        # self.temperature is not needed for Cosine Loss, but we keep it to avoid crashing if passed
        self.ignore_index = ignore_index

        # Cross-entropy loss with ignore_index for undefined PlantSeg classes
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

        # BCE with logits for segmentation
        self.seg_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        student_features: torch.Tensor,
        student_masks: torch.Tensor,
        labels: torch.Tensor,
        teacher_features: torch.Tensor,
        ground_truth_masks: torch.Tensor
    ) -> dict:
        """
        Compute composite loss.
        """
        device = student_logits.device

        # 1. L_ce: Cross-entropy classification loss
        l_ce = self.ce_loss(student_logits, labels)

        # 2. L_dist: Cosine Similarity distillation loss (only for valid labels)
        l_dist = self._compute_distillation_loss(student_features, teacher_features, labels, device)

        # 3. L_seg: Segmentation loss (only for valid masks)
        l_seg = self._compute_segmentation_loss(student_masks, ground_truth_masks, device)

        # Total loss
        total_loss = l_ce + self.alpha * l_dist + self.beta * l_seg

        return {
            'total': total_loss,
            'ce': l_ce,
            'dist': l_dist,
            'seg': l_seg
        }

    def _compute_distillation_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        labels: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Compute Cosine Embedding Loss between student and teacher features.
        Formula: Loss = 1 - CosineSimilarity(Student, Teacher)
        Range: [0, 2], where 0 means identical direction.
        """
        # Find valid samples (label != ignore_index)
        valid_mask = labels != self.ignore_index

        # If no valid samples in batch, return 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=device)

        # Get valid student features [N_valid, 512]
        valid_student_features = student_features[valid_mask]

        # Get corresponding teacher features
        if teacher_features.shape[0] == student_features.shape[0]:
            # Teacher features are per-sample [B, D]
            valid_teacher_features = teacher_features[valid_mask]
        else:
            # Teacher features are per-class [num_classes, D], index by label
            valid_labels = labels[valid_mask]
            valid_teacher_features = teacher_features[valid_labels]

        # --- THE FIX: Use Cosine Similarity instead of KL Divergence ---
        # 1. Compute cosine similarity along the feature dimension (dim=-1)
        # Output shape: [N_valid] with values between -1 and 1
        similarity = F.cosine_similarity(valid_student_features, valid_teacher_features, dim=-1)

        # 2. Convert to loss. We want to MAXIMIZE similarity, so we MINIMIZE (1 - similarity).
        # Loss range: 0 (perfect alignment) to 2 (perfectly opposite)
        l_dist = (1.0 - similarity).mean()

        return l_dist

    def _compute_segmentation_loss(
        self,
        student_masks: torch.Tensor,
        ground_truth_masks: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        # Ensure consistent shape [B, H, W]
        if student_masks.dim() == 4:
            student_masks = student_masks.squeeze(1)
        if ground_truth_masks.dim() == 4:
            ground_truth_masks = ground_truth_masks.squeeze(1)

        # Find valid masks (not all zeros)
        mask_sums = ground_truth_masks.view(ground_truth_masks.shape[0], -1).sum(dim=1)
        valid_mask = mask_sums > 0

        # If no valid masks in batch, return 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=device)

        # Get valid masks
        valid_student_masks = student_masks[valid_mask]
        valid_gt_masks = ground_truth_masks[valid_mask]

        # Compute BCE loss
        l_seg = self.seg_loss(valid_student_masks, valid_gt_masks.float())

        return l_seg

