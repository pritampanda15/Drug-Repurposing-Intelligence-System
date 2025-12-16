"""
Link prediction model combining R-GCN encoder with decoder.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class LinkPredictor(nn.Module):
    """
    Link prediction model for drug-disease relationships.
    """

    def __init__(self, hidden_dim: int = 128):
        """
        Initialize link predictor.

        Parameters
        ----------
        hidden_dim : int
            Hidden dimension size
        """
        super().__init__()

        # Simple MLP decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        z_src: torch.Tensor,
        z_dst: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict link probability.

        Parameters
        ----------
        z_src : torch.Tensor
            Source node embeddings
        z_dst : torch.Tensor
            Destination node embeddings

        Returns
        -------
        torch.Tensor
            Link prediction scores
        """
        # Concatenate embeddings
        z = torch.cat([z_src, z_dst], dim=-1)

        # Decode
        out = self.decoder(z)

        return out.squeeze(-1)


class DrugRepurposingModel(nn.Module):
    """
    Complete model for drug repurposing prediction.
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
        """Initialize complete model."""
        super().__init__()

        # Simplified model for now - use embeddings
        self.compound_embed = nn.Embedding(
            num_nodes_dict.get('Compound', 1000),
            hidden_dim
        )
        self.disease_embed = nn.Embedding(
            num_nodes_dict.get('Disease', 1000),
            hidden_dim
        )

        # Link predictor
        self.link_predictor = LinkPredictor(hidden_dim)

    def forward(
        self,
        compound_idx: torch.Tensor,
        disease_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for drug-disease pairs.

        Parameters
        ----------
        compound_idx : torch.Tensor
            Compound node indices
        disease_idx : torch.Tensor
            Disease node indices

        Returns
        -------
        torch.Tensor
            Prediction scores
        """
        z_compound = self.compound_embed(compound_idx)
        z_disease = self.disease_embed(disease_idx)

        return self.link_predictor(z_compound, z_disease)
