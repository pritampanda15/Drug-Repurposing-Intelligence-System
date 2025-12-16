"""
Graph Builder for Drug Repurposing System

This module builds PyTorch Geometric graphs from Hetionet data
and prepares them for R-GCN training.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

import torch
from torch_geometric.data import HeteroData
import numpy as np
import yaml

from .hetionet_loader import HetionetLoader


logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Build and prepare knowledge graphs for training.
    """

    def __init__(
        self,
        data_dir: str = "data/raw/hetionet",
        config_path: Optional[str] = None
    ):
        """Initialize GraphBuilder."""
        self.data_dir = Path(data_dir)
        self.loader = HetionetLoader(data_dir=str(self.data_dir))
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = "config/model_config.yaml"

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return {
                'model': {'encoder': {'hidden_dim': 128}},
                'data': {'seed': 42}
            }

    def build(
        self,
        add_node_features: bool = True,
        make_undirected: bool = True
    ) -> HeteroData:
        """Build complete knowledge graph with features."""
        logger.info("Building knowledge graph...")
        graph = self.loader.load()

        if add_node_features:
            graph = self._add_node_features(graph)

        if make_undirected:
            graph = self._make_undirected(graph)

        logger.info("Knowledge graph built successfully")
        return graph

    def _add_node_features(self, graph: HeteroData) -> HeteroData:
        """Add initial node features to the graph."""
        logger.info("Adding node features...")
        feature_dim = self.config.get('model', {}).get('encoder', {}).get('hidden_dim', 128)

        for node_type in graph.node_types:
            num_nodes = graph[node_type].num_nodes
            features = torch.randn(num_nodes, feature_dim) * np.sqrt(2.0 / (num_nodes + feature_dim))
            graph[node_type].x = features
            logger.info(f"  {node_type}: {num_nodes} nodes, {feature_dim}D features")

        return graph

    def _make_undirected(self, graph: HeteroData) -> HeteroData:
        """Add reverse edges to make the graph undirected."""
        logger.info("Adding reverse edges...")
        edge_types_to_add = []

        for edge_type in graph.edge_types:
            src_type, rel, tgt_type = edge_type
            reverse_type = (tgt_type, f"rev_{rel}", src_type)
            if reverse_type in graph.edge_types:
                continue

            edge_index = graph[edge_type].edge_index
            reverse_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
            edge_types_to_add.append((reverse_type, reverse_edge_index))

        for edge_type, edge_index in edge_types_to_add:
            graph[edge_type].edge_index = edge_index

        logger.info(f"Added {len(edge_types_to_add)} reverse edge types")
        return graph

    def save_graph(self, graph: HeteroData, output_path: str) -> None:
        """Save graph to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(graph, output_path)
        logger.info(f"Graph saved to {output_path}")

    def print_graph_statistics(self, graph: HeteroData) -> None:
        """Print detailed graph statistics."""
        logger.info("\n" + "=" * 60)
        logger.info("GRAPH STATISTICS")
        logger.info("=" * 60)

        logger.info(f"\nNode Types: {len(graph.node_types)}")
        for node_type in graph.node_types:
            num_nodes = graph[node_type].num_nodes
            has_features = hasattr(graph[node_type], 'x')
            feature_dim = graph[node_type].x.size(1) if has_features else 0
            logger.info(f"  {node_type}: {num_nodes} nodes, {feature_dim}D features")

        logger.info(f"\nEdge Types: {len(graph.edge_types)}")
        total_edges = 0
        for edge_type in graph.edge_types:
            num_edges = graph[edge_type].edge_index.size(1)
            total_edges += num_edges
            src, rel, tgt = edge_type
            logger.info(f"  ({src}, {rel}, {tgt}): {num_edges} edges")

        logger.info(f"\nTotal Nodes: {sum(graph[nt].num_nodes for nt in graph.node_types)}")
        logger.info(f"Total Edges: {total_edges}")
        logger.info("=" * 60 + "\n")
