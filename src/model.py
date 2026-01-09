import torch
import torch.nn as nn
import timm


class DistilledMobileNet(nn.Module):
    def __init__(self, num_classes=89, use_seg_head=False):
        super(DistilledMobileNet, self).__init__()

        # Load MobileNetV4 Small pretrained on ImageNet [cite: 25, 28]
        # "mobilenetv4_conv_small.e2400_r224_in1k" corresponds to the V4 Small variant
        self.backbone = timm.create_model(
            'mobilenetv4_conv_small.e2400_r224_in1k',
            pretrained=True,
            num_classes=0,  # Remove the default head
            global_pool=''  # Remove default pooling
        )

        # Get the actual number of features by running a dummy forward pass
        # MobileNetV4 Small actually outputs 1280 channels, not 960
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            dummy_output = self.backbone(dummy_input)
            self.num_features = dummy_output.shape[1]  # Get actual channel dimension
            self.feature_map_size = dummy_output.shape[2]  # Get spatial dimension (e.g., 7)

        print(f"MobileNetV4 backbone output features: {self.num_features}, feature map: {self.feature_map_size}x{self.feature_map_size}")

        # 1. Classification Head (The Baseline) [cite: 25]
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_features, num_classes)
        )

        # 2. Projection Head (For CLIP Distillation)
        # We need to project image features to match CLIP's text embedding dimension (usually 512)
        self.distill_projector = nn.Linear(self.num_features, 512)

        # 3. Segmentation Head (Optional Phase 3) [cite: 26]
        self.use_seg_head = use_seg_head
        if self.use_seg_head:
            # A simple lightweight decoder (U-Net Lite style)
            # Use size=(224, 224) instead of scale_factor for reliable upsampling
            self.seg_head = nn.Sequential(
                nn.Conv2d(self.num_features, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 1, kernel_size=1)  # Binary mask output
            )
        else:
            self.seg_head = None

    def forward(self, x):
        input_size = x.shape[-2:]  # (H, W) of input image

        # Extract features from MobileNetV4 backbone
        features = self.backbone(x)  # Shape: [Batch, 1280, 7, 7]

        # Classification path
        pooled = self.global_pool(features)
        logits = self.classifier(pooled)

        # Distillation path (Project features to align with CLIP)
        # We flatten the pooled features for the projector
        flat_features = pooled.flatten(1)
        distill_tokens = self.distill_projector(flat_features)

        # Segmentation path
        if self.use_seg_head and self.seg_head is not None:
            mask_logits = self.seg_head(features)
            # Upsample to match input image size
            mask_logits = nn.functional.interpolate(
                mask_logits, size=input_size, mode='bilinear', align_corners=False
            )
            mask_logits = mask_logits.squeeze(1)  # [B, H, W]
        else:
            # Return zeros with correct shape when seg_head is disabled
            batch_size = x.shape[0]
            mask_logits = torch.zeros(batch_size, input_size[0], input_size[1], device=x.device)

        return logits, distill_tokens, mask_logits