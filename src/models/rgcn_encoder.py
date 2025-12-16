"""
Relational Graph Convolutional Network (R-GCN) Encoder

Implementation of R-GCN for heterogeneous graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, Linear
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple


class RGCNEncoder(nn.Module):
    """
    R-GCN encoder for heterogeneous biomedical knowledge graphs.

    Uses relation-specific transformations to handle multiple edge types.
    """

    def __init__(
        self,
        num_nodes_dict: Dict[str, int],
        num_relations: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_bases: int = 30,
        dropout: float = 0.3
    ):
        """
        Initialize R-GCN encoder.

        Parameters
        ----------
        num_nodes_dict : Dict[str, int]
            Number of nodes per node type
        num_relations : int
            Number of relation types
        hidden_dim : int
            Hidden dimension size
        num_layers : int
            Number of R-GCN layers
        num_bases : int
            Number of bases for basis decomposition
        dropout : float
            Dropout rate
        """
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Node type embeddings
        self.node_embeddings = nn.ModuleDict({
            node_type: nn.Embedding(num_nodes, hidden_dim)
            for node_type, num_nodes in num_nodes_dict.items()
        })

        # R-GCN layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                RGCNConv(
                    hidden_dim,
                    hidden_dim,
                    num_relations=num_relations,
                    num_bases=num_bases
                )
            )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x_dict : Dict[str, torch.Tensor]
            Node features per type
        edge_index_dict : Dict[Tuple, torch.Tensor]
            Edge indices per edge type

        Returns
        -------
        Dict[str, torch.Tensor]
            Node embeddings per type
        """
        # Get embeddings
        for node_type in x_dict.keys():
            if node_type in self.node_embeddings:
                num_nodes = x_dict[node_type].size(0) if x_dict[node_type].ndim > 0 else 1
                x_dict[node_type] = self.node_embeddings[node_type].weight[:num_nodes]

        # Apply R-GCN layers
        for i, conv in enumerate(self.convs):
            x_dict_new = {}

            for node_type in x_dict.keys():
                x_dict_new[node_type] = torch.zeros_like(x_dict[node_type])

            # Apply convolution
            for edge_type, edge_index in edge_index_dict.items():
                src_type, rel_type, dst_type = edge_type

                if src_type in x_dict and dst_type in x_dict_new:
                    # Simplified: using first relation for demo
                    # In full implementation, would use relation index
                    out = self.convs[i](
                        x_dict[src_type],
                        edge_index,
                        edge_type=0  # Simplified
                    )

            # Activation and dropout
            for node_type in x_dict_new.keys():
                x_dict_new[node_type] = F.relu(x_dict_new[node_type])
                if i < self.num_layers - 1:
                    x_dict_new[node_type] = F.dropout(
                        x_dict_new[node_type],
                        p=self.dropout,
                        training=self.training
                    )

            x_dict = x_dict_new

        return x_dict
