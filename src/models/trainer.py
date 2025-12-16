"""
Training logic for drug repurposing model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional, Tuple
import logging
from tqdm import tqdm

from ..utils.metrics import compute_metrics


logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for drug repurposing model.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        """
        Initialize trainer.

        Parameters
        ----------
        model : nn.Module
            Model to train
        device : str
            Device to train on
        learning_rate : float
            Learning rate
        weight_decay : float
            Weight decay for regularization
        """
        self.model = model.to(device)
        self.device = device

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.criterion = nn.BCEWithLogitsLoss()

    def train_epoch(
        self,
        train_loader: DataLoader
    ) -> float:
        """
        Train for one epoch.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader

        Returns
        -------
        float
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(train_loader, desc="Training"):
            compound_idx, disease_idx, labels = [b.to(self.device) for b in batch]

            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(compound_idx, disease_idx)

            # Compute loss
            loss = self.criterion(logits, labels.float())

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate model.

        Parameters
        ----------
        val_loader : DataLoader
            Validation data loader

        Returns
        -------
        Tuple[float, Dict[str, float]]
            Average loss and metrics dictionary
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        num_batches = 0

        for batch in val_loader:
            compound_idx, disease_idx, labels = [b.to(self.device) for b in batch]

            # Forward pass
            logits = self.model(compound_idx, disease_idx)

            # Compute loss
            loss = self.criterion(logits, labels.float())

            total_loss += loss.item()
            num_batches += 1

            # Store predictions
            preds = torch.sigmoid(logits)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Compute metrics
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        metrics = {
            'auc': 0.0,  # Would compute properly with sklearn
            'avg_pred': all_preds.mean().item()
        }

        return avg_loss, metrics

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Checkpoint loaded from {path}")
