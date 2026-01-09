import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_loader import get_domain_dataloaders
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

    for batch in pbar:
        # Move data to device
        images = batch[0].to(device)
        labels = batch[1].to(device)

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
    epoch: int = 0,
    desc: str = "Val"
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

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [{desc}]")

    for batch in pbar:
        # Move data to device
        images = batch[0].to(device)
        labels = batch[1].to(device)

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

    # Initialize dataset and dataloaders via Domain Loader
    print("\nLoading datasets (Source: PlantVillage, Target: PlantDoc/Wild)...")
    
    dataloaders, class_to_idx = get_domain_dataloaders(
        mapping_csv=config['data']['csv_path'],
        root_dir="data",  # Assumes data is in root/data
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        val_split=0.2  # Default split
    )
    
    train_loader = dataloaders['source_train']
    val_loader = dataloaders['source_val']
    target_loader = dataloaders['target_test']

    print(f"Classes: {len(class_to_idx)}")
    print(f"Source Train batches: {len(train_loader)}")
    print(f"Source Val batches:   {len(val_loader)}")
    print(f"Target Test batches:  {len(target_loader)}")

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

        # Train on Source
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch
        )
        print(f"Train Source Loss: {train_loss:.4f}")

        # Validate on Source
        val_acc, val_loss = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            desc="Src Val"
        )
        print(f"Val Source - Acc: {val_acc:.4f}, Loss: {val_loss:.4f}")

        # Validate on Target (The real test!)
        target_acc, target_loss = validate(
            model=model,
            dataloader=target_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            desc="Tgt Test"
        )
        print(f"Test Target - Acc: {target_acc:.4f}, Loss: {target_loss:.4f}")

        # Save best model (based on Source Validation for now, standard practice)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'target_accuracy': target_acc,
                'config': config
            }, best_model_path)
            print(f"*** New best model saved! Source Acc: {val_acc:.4f} ***")

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best Source Validation Accuracy: {best_val_acc:.4f}")
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