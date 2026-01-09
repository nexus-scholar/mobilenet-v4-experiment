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


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int = 0
) -> float:
    """
    Train for one epoch.

    Returns:
        Average loss for the epoch
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")

    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        logits = model(images)
        loss = criterion(logits, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    return running_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int = 0
) -> tuple:
    """
    Validate the model.

    Returns:
        tuple: (accuracy, average_loss)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]")

    for batch in pbar:
        # Move data to device
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item()
        num_batches += 1

        # Calculate accuracy
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        acc = correct / total if total > 0 else 0
        pbar.set_postfix({'acc': f"{acc:.4f}"})

    avg_loss = running_loss / max(num_batches, 1)
    accuracy = correct / total if total > 0 else 0
    return accuracy, avg_loss


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
        num_classes=config['model']['num_classes']
    )
    model = model.to(device)

    # Initialize loss function
    criterion = nn.CrossEntropyLoss(ignore_index=config['training']['ignore_index'])

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

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
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch
        )
        print(f"Train - Loss: {train_loss:.4f}")

        # Validate
        val_acc, val_loss = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch
        )
        print(f"Val - Accuracy: {val_acc:.4f}, Loss: {val_loss:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'val_loss': val_loss,
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
