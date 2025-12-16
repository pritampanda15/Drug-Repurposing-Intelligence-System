#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train Drug Repurposing Model

This script trains an R-GCN model for predicting drug-disease relationships.

Usage:
    python scripts/train_model.py [OPTIONS]
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.hetionet_loader import HetionetLoader
from src.models.link_predictor import DrugRepurposingModel
from src.models.trainer import Trainer
from src.utils.logging_config import setup_logging


logger = logging.getLogger(__name__)


def prepare_data(loader: HetionetLoader, negative_ratio: int = 5):
    """Prepare training data."""
    logger.info("Preparing training data...")

    # Get positive pairs
    positive_pairs = loader.get_drug_disease_pairs()
    logger.info(f"Positive pairs: {len(positive_pairs)}")

    # Get negative pairs
    num_negatives = len(positive_pairs) * negative_ratio
    negative_pairs = loader.build_negative_samples(num_negatives)
    logger.info(f"Negative pairs: {len(negative_pairs)}")

    # Convert to tensors
    compound_mapping = loader.node_mappings['Compound']
    disease_mapping = loader.node_mappings['Disease']

    # Positive examples
    pos_compound_idx = torch.tensor([
        compound_mapping[row['drug_id']]
        for _, row in positive_pairs.iterrows()
    ], dtype=torch.long)

    pos_disease_idx = torch.tensor([
        disease_mapping[row['disease_id']]
        for _, row in positive_pairs.iterrows()
    ], dtype=torch.long)

    pos_labels = torch.ones(len(positive_pairs))

    # Negative examples
    neg_compound_idx = torch.tensor([
        compound_mapping[row['drug_id']]
        for _, row in negative_pairs.iterrows()
    ], dtype=torch.long)

    neg_disease_idx = torch.tensor([
        disease_mapping[row['disease_id']]
        for _, row in negative_pairs.iterrows()
    ], dtype=torch.long)

    neg_labels = torch.zeros(len(negative_pairs))

    # Combine
    compound_idx = torch.cat([pos_compound_idx, neg_compound_idx])
    disease_idx = torch.cat([pos_disease_idx, neg_disease_idx])
    labels = torch.cat([pos_labels, neg_labels])

    # Shuffle
    perm = torch.randperm(len(labels))
    compound_idx = compound_idx[perm]
    disease_idx = disease_idx[perm]
    labels = labels[perm]

    # Split into train/val/test
    n = len(labels)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    train_dataset = TensorDataset(
        compound_idx[:train_end],
        disease_idx[:train_end],
        labels[:train_end]
    )

    val_dataset = TensorDataset(
        compound_idx[train_end:val_end],
        disease_idx[train_end:val_end],
        labels[train_end:val_end]
    )

    test_dataset = TensorDataset(
        compound_idx[val_end:],
        disease_idx[val_end:],
        labels[val_end:]
    )

    return train_dataset, val_dataset, test_dataset


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train drug repurposing model"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/model_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw/hetionet',
        help='Directory containing Hetionet data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/checkpoints/best_model.pt',
        help='Output path for trained model'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=512,
        help='Batch size'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level="INFO", log_file="logs/training/train.log")

    logger.info("=" * 60)
    logger.info("Training Drug Repurposing Model")
    logger.info("=" * 60)

    try:
        # Load configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        # Load Hetionet
        loader = HetionetLoader(data_dir=args.data_dir)
        loader.load()

        metadata = loader.get_metadata()
        logger.info(f"Loaded graph with {metadata['total_nodes']} nodes and {metadata['total_edges']} edges")

        # Prepare data
        train_dataset, val_dataset, test_dataset = prepare_data(loader)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

        logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

        # Initialize model
        num_nodes_dict = {k: v for k, v in metadata['node_types'].items()}

        model = DrugRepurposingModel(
            num_nodes_dict=num_nodes_dict,
            num_relations=metadata['num_edge_types'],
            hidden_dim=int(config['model']['encoder']['hidden_dim']),
            num_layers=int(config['model']['encoder']['num_layers'])
        )

        logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

        # Initialize trainer
        trainer = Trainer(
            model=model,
            device=args.device,
            learning_rate=float(config['training']['learning_rate']),
            weight_decay=float(config['training']['weight_decay'])
        )

        # Training loop
        best_val_loss = float('inf')

        for epoch in range(1, args.epochs + 1):
            logger.info(f"\nEpoch {epoch}/{args.epochs}")

            # Train
            train_loss = trainer.train_epoch(train_loader)

            # Validate
            val_loss, val_metrics = trainer.evaluate(val_loader)

            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                trainer.save_checkpoint(args.output)
                logger.info(f"✓ New best model saved (val_loss: {val_loss:.4f})")

        # Test evaluation
        logger.info("\nEvaluating on test set...")
        test_loss, test_metrics = trainer.evaluate(test_loader)
        logger.info(f"Test Loss: {test_loss:.4f}")

        logger.info("\n" + "=" * 60)
        logger.info("✓ Training complete!")
        logger.info(f"Best model saved to: {args.output}")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
