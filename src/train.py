import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from src.dataset import PlantDiseaseDataset
from src.model import DistilledMobileNet
from src.loss import CompositeLoss
from src import teacher as teacher_module


def load_class_names(classes_path: str = "data/PlantWild/classes.txt") -> list:
    """Load class names from classes.txt file."""
    class_names = []
    with open(classes_path, 'r') as f:
        for line in f:
            # Format: "0 apple black rot"
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                class_names.append(parts[1])
    return class_names


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: CompositeLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    teacher_model: nn.Module = None,
    teacher_tokenizer = None,
    class_text_features: torch.Tensor = None,
    use_distillation: bool = False,
    epoch: int = 0
) -> dict:
    """
    Train for one epoch.

    Returns:
        Dictionary with average losses for the epoch
    """
    model.train()

    running_loss = {'total': 0.0, 'ce': 0.0, 'dist': 0.0, 'seg': 0.0}
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")

    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        masks = batch['mask'].to(device) if batch['mask'] is not None else None

        # Forward pass through student
        logits, features, mask_logits = model(images)

        # Get teacher features if using distillation
        if use_distillation and class_text_features is not None:
            # Use pre-computed class text features indexed by labels
            # For samples with label=-1, the distillation loss will be 0 anyway
            teacher_features = class_text_features
        else:
            # Create dummy teacher features (won't be used since alpha=0)
            teacher_features = torch.zeros_like(features)

        # Prepare ground truth masks
        if masks is None:
            gt_masks = torch.zeros_like(mask_logits)
        else:
            # Resize masks to match mask_logits if needed
            if masks.shape[-2:] != mask_logits.shape[-2:]:
                gt_masks = nn.functional.interpolate(
                    masks.unsqueeze(1).float(),
                    size=mask_logits.shape[-2:],
                    mode='nearest'
                ).squeeze(1)
            else:
                gt_masks = masks.float()

        # Compute loss
        losses = criterion(
            student_logits=logits,
            student_features=features,
            student_masks=mask_logits,
            labels=labels,
            teacher_features=teacher_features,
            ground_truth_masks=gt_masks
        )

        # Backpropagation
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()

        # Accumulate losses
        for key in running_loss:
            running_loss[key] += losses[key].item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{losses['total'].item():.4f}",
            'ce': f"{losses['ce'].item():.4f}"
        })

        # Log every 100 batches
        if (batch_idx + 1) % 100 == 0:
            avg_loss = running_loss['total'] / num_batches
            avg_ce = running_loss['ce'] / num_batches
            avg_dist = running_loss['dist'] / num_batches
            avg_seg = running_loss['seg'] / num_batches
            print(f"\n  Batch {batch_idx+1}: Loss={avg_loss:.4f} "
                  f"(CE={avg_ce:.4f}, Dist={avg_dist:.4f}, Seg={avg_seg:.4f})")

    # Compute epoch averages
    avg_losses = {key: val / num_batches for key, val in running_loss.items()}
    return avg_losses


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: CompositeLoss,
    device: torch.device,
    class_text_features: torch.Tensor = None,
    use_distillation: bool = False,
    epoch: int = 0
) -> tuple:
    """
    Validate the model.

    Important: Accuracy is only calculated on samples where label != -1.

    Returns:
        tuple: (accuracy, average_losses_dict)
    """
    model.eval()

    running_loss = {'total': 0.0, 'ce': 0.0, 'dist': 0.0, 'seg': 0.0}
    correct = 0
    total_valid = 0  # Only count samples with valid labels
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]")

    for batch in pbar:
        # Move data to device
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        masks = batch['mask'].to(device) if batch['mask'] is not None else None

        # Forward pass
        logits, features, mask_logits = model(images)

        # Get teacher features
        if use_distillation and class_text_features is not None:
            teacher_features = class_text_features
        else:
            teacher_features = torch.zeros_like(features)

        # Prepare ground truth masks
        if masks is None:
            gt_masks = torch.zeros_like(mask_logits)
        else:
            if masks.shape[-2:] != mask_logits.shape[-2:]:
                gt_masks = nn.functional.interpolate(
                    masks.unsqueeze(1).float(),
                    size=mask_logits.shape[-2:],
                    mode='nearest'
                ).squeeze(1)
            else:
                gt_masks = masks.float()

        # Compute loss
        losses = criterion(
            student_logits=logits,
            student_features=features,
            student_masks=mask_logits,
            labels=labels,
            teacher_features=teacher_features,
            ground_truth_masks=gt_masks
        )

        # Accumulate losses
        for key in running_loss:
            running_loss[key] += losses[key].item()
        num_batches += 1

        # Calculate accuracy only on valid labels (label != -1)
        valid_mask = labels != -1
        if valid_mask.any():
            valid_logits = logits[valid_mask]
            valid_labels = labels[valid_mask]
            predictions = valid_logits.argmax(dim=1)
            correct += (predictions == valid_labels).sum().item()
            total_valid += valid_mask.sum().item()

        # Update progress bar
        current_acc = correct / total_valid if total_valid > 0 else 0
        pbar.set_postfix({'acc': f"{current_acc:.4f}"})

    # Compute averages
    accuracy = correct / total_valid if total_valid > 0 else 0
    avg_losses = {key: val / num_batches for key, val in running_loss.items()}

    return accuracy, avg_losses


def main(config_path: str = "config.yaml"):
    # Load configuration
    config = load_config(config_path)
    print(f"Experiment: {config['experiment_name']}")
    print(f"Config: {config}")

    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directories
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Initialize dataset and dataloaders
    print("\nLoading datasets...")

    # Define transforms for training and validation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load full dataset first to split
    full_dataset = PlantDiseaseDataset(
        csv_file=config['data']['csv_path'],
        transform=train_transform
    )

    # Split into train/val (80/20)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Note: val_dataset will use train_transform, but for baseline this is acceptable
    # For more accurate validation, you would need a separate dataset instance

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data'].get('pin_memory', False)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data'].get('pin_memory', False)
    )
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Initialize student model
    print("\nInitializing student model (DistilledMobileNet)...")
    model = DistilledMobileNet(
        num_classes=config['model']['num_classes'],
        use_seg_head=config['model'].get('use_segmentation', False)
    )
    model = model.to(device)

    # Initialize loss function
    criterion = CompositeLoss(
        alpha=config['training']['alpha'],
        beta=config['training']['beta'],
        temperature=config['training'].get('temperature', 1.0),
        ignore_index=config['training']['ignore_index']
    )

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # Load CLIP teacher if using distillation (alpha > 0)
    use_distillation = config['training']['alpha'] > 0
    teacher_model = None
    teacher_tokenizer = None
    class_text_features = None

    if use_distillation:
        print("\nLoading CLIP teacher model for distillation...")
        teacher_model, teacher_tokenizer = teacher_module.load_clip_model()
        teacher_model = teacher_model.to(device)

        # Pre-compute text features for all classes
        class_names = load_class_names()
        print(f"Encoding {len(class_names)} class prompts...")
        class_text_features = teacher_module.encode_class_prompts(
            teacher_model, teacher_tokenizer, class_names, device
        )
        print(f"Class text features shape: {class_text_features.shape}")
    else:
        print("\nSkipping CLIP teacher (alpha=0, Baseline Phase)")

    # Training loop
    print(f"\nStarting training for {config['training']['epochs']} epochs...")
    best_val_acc = 0.0

    # Determine model save path based on experiment name
    experiment_name = config['experiment_name'].lower().replace(' ', '_')
    best_model_path = f"models/{experiment_name}_best_model.pth"
    print(f"Best model will be saved to: {best_model_path}")

    for epoch in range(config['training']['epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['training']['epochs']}")
        print('='*60)

        # Train
        train_losses = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            teacher_model=teacher_model,
            teacher_tokenizer=teacher_tokenizer,
            class_text_features=class_text_features,
            use_distillation=use_distillation,
            epoch=epoch
        )
        print(f"\nTrain - Loss: {train_losses['total']:.4f} "
              f"(CE: {train_losses['ce']:.4f}, Dist: {train_losses['dist']:.4f}, Seg: {train_losses['seg']:.4f})")

        # Validate
        val_acc, val_losses = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            class_text_features=class_text_features,
            use_distillation=use_distillation,
            epoch=epoch
        )
        print(f"Val - Accuracy: {val_acc:.4f}, Loss: {val_losses['total']:.4f} "
              f"(CE: {val_losses['ce']:.4f}, Dist: {val_losses['dist']:.4f}, Seg: {val_losses['seg']:.4f})")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'val_loss': val_losses['total'],
                'config': config
            }, best_model_path)
            print(f"*** New best model saved! Accuracy: {val_acc:.4f} ***")

    print(f"\n{'='*60}")
    print(f"Training complete! Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model saved to: {best_model_path}")
    print('='*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DistilledMobileNet model for plant disease classification")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file (default: config.yaml)"
    )
    args = parser.parse_args()

    main(config_path=args.config)

