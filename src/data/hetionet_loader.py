"""
Hetionet Knowledge Graph Loader

This module handles downloading, parsing, and converting the Hetionet
biomedical knowledge graph into PyTorch Geometric format for R-GCN training.

Hetionet contains drug-gene-disease relationships essential for
drug repurposing prediction via link prediction.

Classes
-------
HetionetLoader
    Main class for loading and processing Hetionet data

Functions
---------
download_hetionet(data_dir: Path) -> Path
    Download Hetionet JSON from official repository
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import json
import logging
import pickle

import torch
from torch_geometric.data import HeteroData
import pandas as pd
import networkx as nx
import numpy as np
from tqdm import tqdm


class HetionetLoader:
    """
    Load and process Hetionet knowledge graph for drug repurposing.

    This class downloads the Hetionet graph, parses node and edge data,
    and converts it to PyTorch Geometric HeteroData format suitable
    for training relational graph neural networks.

    Attributes
    ----------
    data_dir : Path
        Directory for storing raw and processed data
    graph : HeteroData
        PyTorch Geometric heterogeneous graph object
    node_mappings : Dict[str, Dict[str, int]]
        Mapping from node identifiers to integer indices per node type
    edge_type_map : Dict[str, int]
        Mapping from edge type strings to integer indices

    Methods
    -------
    load() -> HeteroData
        Load or download Hetionet and return as HeteroData
    get_drug_disease_pairs() -> pd.DataFrame
        Extract all compound-treats-disease edges for training
    get_node_features(node_type: str) -> torch.Tensor
        Generate initial node features for specified type
    build_negative_samples(num_samples: int) -> torch.Tensor
        Generate negative drug-disease pairs for training
    """

    HETIONET_URL = "https://github.com/hetio/hetionet/raw/main/hetnet/json/hetionet-v1.0.json.bz2"

    NODE_TYPES = [
        "Compound", "Disease", "Gene", "Anatomy", "Pathway",
        "Side Effect", "Pharmacologic Class", "Biological Process",
        "Cellular Component", "Molecular Function", "Symptom"
    ]

    EDGE_TYPES = [
        ("Compound", "treats", "Disease"),
        ("Compound", "palliates", "Disease"),
        ("Compound", "targets", "Gene"),
        ("Compound", "causes", "Side Effect"),
        ("Disease", "associates", "Gene"),
        ("Disease", "localizes", "Anatomy"),
        ("Gene", "participates", "Pathway"),
        # Additional edge types loaded dynamically
    ]

    def __init__(self, data_dir: str = "data/raw/hetionet"):
        """
        Initialize HetionetLoader.

        Parameters
        ----------
        data_dir : str
            Path to directory for Hetionet data storage
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.graph: Optional[HeteroData] = None
        self.node_mappings: Dict[str, Dict[str, int]] = {}
        self.edge_type_map: Dict[str, int] = {}
        self.hetionet_data: Optional[Dict] = None
        self.nodes_by_type: Dict[str, List[Dict]] = defaultdict(list)
        self.edges_by_type: Dict[Tuple[str, str, str], List[Tuple]] = defaultdict(list)
        self.logger = logging.getLogger(__name__)

    def load(self) -> HeteroData:
        """
        Load Hetionet and convert to PyTorch Geometric HeteroData.

        Returns
        -------
        HeteroData
            Heterogeneous graph with node features and edge indices
        """
        # Check for cached processed data
        cache_file = self.data_dir / "hetionet_processed.pkl"
        if cache_file.exists():
            self.logger.info(f"Loading cached Hetionet from {cache_file}")
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
                self.hetionet_data = cached['raw_data']
                self.node_mappings = cached['node_mappings']
                self.nodes_by_type = cached['nodes_by_type']
                self.edges_by_type = cached['edges_by_type']
            self.logger.info("Cached data loaded successfully")
        else:
            # Load raw JSON
            self._load_raw_json()
            # Parse and organize data
            self._parse_nodes()
            self._parse_edges()
            # Cache the processed data
            self._cache_data(cache_file)

        # Convert to PyTorch Geometric HeteroData
        self.graph = self._build_hetero_data()

        return self.graph

    def _load_raw_json(self) -> None:
        """Load Hetionet JSON file."""
        json_file = self.data_dir / "hetionet-v1.0.json"

        if not json_file.exists():
            raise FileNotFoundError(
                f"Hetionet JSON not found at {json_file}. "
                "Run scripts/download_data.py first."
            )

        self.logger.info(f"Loading Hetionet from {json_file}")
        with open(json_file, 'r') as f:
            self.hetionet_data = json.load(f)

        self.logger.info(
            f"Loaded {len(self.hetionet_data.get('nodes', []))} nodes and "
            f"{len(self.hetionet_data.get('edges', []))} edges"
        )

    def _parse_nodes(self) -> None:
        """Parse and organize nodes by type."""
        self.logger.info("Parsing nodes...")

        for node in tqdm(self.hetionet_data['nodes'], desc="Parsing nodes"):
            node_type = node['kind']
            node_id = str(node['identifier'])  # Convert to string for consistent lookup
            node_name = node.get('name', node_id)

            # Add to type-specific list
            self.nodes_by_type[node_type].append({
                'id': node_id,
                'name': node_name,
                'data': node
            })

            # Create mapping: node_id -> index within type
            if node_type not in self.node_mappings:
                self.node_mappings[node_type] = {}

            self.node_mappings[node_type][node_id] = len(self.node_mappings[node_type])

        # Log statistics
        for node_type, nodes in self.nodes_by_type.items():
            self.logger.info(f"  {node_type}: {len(nodes)} nodes")

    def _parse_edges(self) -> None:
        """Parse and organize edges by type."""
        self.logger.info("Parsing edges...")

        for edge in tqdm(self.hetionet_data['edges'], desc="Parsing edges"):
            # Hetionet format: source_id and target_id are lists [node_type, identifier]
            source_id_list = edge['source_id']
            target_id_list = edge['target_id']
            relation = edge['kind']

            # Extract node type and identifier from lists
            if isinstance(source_id_list, list) and len(source_id_list) == 2:
                source_type = source_id_list[0]
                source_id = str(source_id_list[1])
            else:
                # Fallback for non-list format
                source_id = str(source_id_list)
                source_type = self._get_node_type(source_id)
                if source_type is None:
                    continue

            if isinstance(target_id_list, list) and len(target_id_list) == 2:
                target_type = target_id_list[0]
                target_id = str(target_id_list[1])
            else:
                # Fallback for non-list format
                target_id = str(target_id_list)
                target_type = self._get_node_type(target_id)
                if target_type is None:
                    continue

            edge_type = (source_type, relation, target_type)

            # Add edge (as node indices)
            if source_type in self.node_mappings and target_type in self.node_mappings:
                if source_id in self.node_mappings[source_type] and target_id in self.node_mappings[target_type]:
                    source_idx = self.node_mappings[source_type][source_id]
                    target_idx = self.node_mappings[target_type][target_id]

                    self.edges_by_type[edge_type].append((source_idx, target_idx))

        # Log statistics
        for edge_type, edges in self.edges_by_type.items():
            src_type, rel, tgt_type = edge_type
            self.logger.info(f"  ({src_type}, {rel}, {tgt_type}): {len(edges)} edges")

    def _get_node_type(self, node_id: str) -> Optional[str]:
        """Get node type from node ID by looking it up."""
        for node_type, mapping in self.node_mappings.items():
            if node_id in mapping:
                return node_type
        # If not in mappings yet, search in raw data
        for node in self.hetionet_data.get('nodes', []):
            if node['identifier'] == node_id:
                return node['kind']
        return None

    def _cache_data(self, cache_file: Path) -> None:
        """Cache processed data for faster loading."""
        self.logger.info(f"Caching processed data to {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'raw_data': self.hetionet_data,
                'node_mappings': self.node_mappings,
                'nodes_by_type': self.nodes_by_type,
                'edges_by_type': self.edges_by_type
            }, f)

    def _build_hetero_data(self) -> HeteroData:
        """Convert parsed data to PyTorch Geometric HeteroData."""
        self.logger.info("Building HeteroData object...")

        data = HeteroData()

        # Add node information
        for node_type, nodes in self.nodes_by_type.items():
            num_nodes = len(nodes)
            # Store number of nodes (features will be added during training)
            data[node_type].num_nodes = num_nodes
            # Store node IDs and names for reference
            data[node_type].node_ids = [n['id'] for n in nodes]
            data[node_type].node_names = [n['name'] for n in nodes]

        # Add edge information
        for edge_type, edges in self.edges_by_type.items():
            if len(edges) > 0:
                edge_index = torch.tensor(edges, dtype=torch.long).t()
                data[edge_type].edge_index = edge_index

        self.logger.info(f"HeteroData created with {len(self.nodes_by_type)} node types "
                        f"and {len(self.edges_by_type)} edge types")

        return data

    def get_drug_disease_pairs(
        self,
        split: str = "all"
    ) -> pd.DataFrame:
        """
        Extract compound-treats-disease relationships.

        Parameters
        ----------
        split : str
            One of "all", "train", "valid", "test"

        Returns
        -------
        pd.DataFrame
            Columns: drug_id, drug_name, disease_id, disease_name, label
        """
        if self.graph is None:
            self.load()

        pairs = []

        # Find treats edges
        for edge_type, edges in self.edges_by_type.items():
            src_type, rel, tgt_type = edge_type
            if src_type == "Compound" and rel == "treats" and tgt_type == "Disease":
                # Get compound and disease info
                compounds = self.nodes_by_type["Compound"]
                diseases = self.nodes_by_type["Disease"]

                for src_idx, tgt_idx in edges:
                    pairs.append({
                        'drug_id': compounds[src_idx]['id'],
                        'drug_name': compounds[src_idx]['name'],
                        'disease_id': diseases[tgt_idx]['id'],
                        'disease_name': diseases[tgt_idx]['name'],
                        'label': 1
                    })

        df = pd.DataFrame(pairs)

        # TODO: Implement train/val/test splits if needed
        if split != "all":
            self.logger.warning(f"Split '{split}' not implemented, returning all data")

        return df

    def get_node_features(
        self,
        node_type: str,
        feature_dim: int = 128
    ) -> torch.Tensor:
        """
        Generate initial node feature matrix for specified node type.

        Parameters
        ----------
        node_type : str
            One of the NODE_TYPES
        feature_dim : int
            Dimension of feature vectors

        Returns
        -------
        torch.Tensor
            Shape (num_nodes, feature_dim)
        """
        if node_type not in self.nodes_by_type:
            raise ValueError(f"Unknown node type: {node_type}")

        num_nodes = len(self.nodes_by_type[node_type])

        # Initialize with random embeddings (will be learned during training)
        # Using Xavier/Glorot initialization
        features = torch.randn(num_nodes, feature_dim) * np.sqrt(2.0 / (num_nodes + feature_dim))

        return features

    def build_negative_samples(
        self,
        num_samples: int,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Generate negative drug-disease pairs for training.

        Parameters
        ----------
        num_samples : int
            Number of negative samples to generate
        seed : int
            Random seed for reproducibility

        Returns
        -------
        pd.DataFrame
            Negative drug-disease pairs
        """
        np.random.seed(seed)

        if "Compound" not in self.nodes_by_type or "Disease" not in self.nodes_by_type:
            raise ValueError("Compound or Disease nodes not found")

        compounds = self.nodes_by_type["Compound"]
        diseases = self.nodes_by_type["Disease"]

        # Get existing positive pairs
        positive_pairs = set()
        for edge_type, edges in self.edges_by_type.items():
            src_type, rel, tgt_type = edge_type
            if src_type == "Compound" and tgt_type == "Disease":
                for src_idx, tgt_idx in edges:
                    positive_pairs.add((src_idx, tgt_idx))

        # Generate negative samples
        negative_pairs = []
        attempts = 0
        max_attempts = num_samples * 10

        while len(negative_pairs) < num_samples and attempts < max_attempts:
            drug_idx = np.random.randint(0, len(compounds))
            disease_idx = np.random.randint(0, len(diseases))

            if (drug_idx, disease_idx) not in positive_pairs:
                negative_pairs.append({
                    'drug_id': compounds[drug_idx]['id'],
                    'drug_name': compounds[drug_idx]['name'],
                    'disease_id': diseases[disease_idx]['id'],
                    'disease_name': diseases[disease_idx]['name'],
                    'label': 0
                })
                positive_pairs.add((drug_idx, disease_idx))  # Avoid duplicates

            attempts += 1

        return pd.DataFrame(negative_pairs)

    def get_metadata(self) -> Dict:
        """
        Get graph metadata and statistics.

        Returns
        -------
        Dict
            Dictionary containing graph statistics
        """
        if self.graph is None:
            self.load()

        metadata = {
            'num_node_types': len(self.nodes_by_type),
            'num_edge_types': len(self.edges_by_type),
            'node_types': {},
            'edge_types': {},
            'total_nodes': sum(len(nodes) for nodes in self.nodes_by_type.values()),
            'total_edges': sum(len(edges) for edges in self.edges_by_type.values())
        }

        for node_type, nodes in self.nodes_by_type.items():
            metadata['node_types'][node_type] = len(nodes)

        for edge_type, edges in self.edges_by_type.items():
            src, rel, tgt = edge_type
            key = f"({src}, {rel}, {tgt})"
            metadata['edge_types'][key] = len(edges)

        return metadata
