#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build Knowledge Graph Script

This script processes Hetionet data and creates a PyTorch Geometric
HeteroData object ready for R-GCN training.

Usage:
    python scripts/build_knowledge_graph.py [OPTIONS]
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.graph_builder import GraphBuilder


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build knowledge graph from Hetionet data"
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
        default='data/processed/graph/hetionet_graph.pt',
        help='Output path for processed graph'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/model_config.yaml',
        help='Path to model configuration file'
    )
    parser.add_argument(
        '--no-features',
        action='store_true',
        help='Do not add node features'
    )
    parser.add_argument(
        '--no-undirected',
        action='store_true',
        help='Do not add reverse edges'
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Building Knowledge Graph for Drug Repurposing")
    logger.info("=" * 60)

    try:
        # Initialize graph builder
        builder = GraphBuilder(
            data_dir=args.data_dir,
            config_path=args.config
        )

        # Build graph
        graph = builder.build(
            add_node_features=not args.no_features,
            make_undirected=not args.no_undirected
        )

        # Print statistics
        builder.print_graph_statistics(graph)

        # Save graph
        builder.save_graph(graph, args.output)

        logger.info("\n" + "=" * 60)
        logger.info("✓ Knowledge graph built successfully!")
        logger.info(f"Saved to: {args.output}")
        logger.info("\nNext steps:")
        logger.info("  1. python scripts/train_model.py")
        logger.info("  2. python scripts/index_documents.py")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Failed to build knowledge graph: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
