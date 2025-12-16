## Drug Repurposing Intelligence System: Planning Document

---

## Section 1: Data and Deep Learning Model Setup

### Recommended Primary Dataset: Hetionet

**Source:** https://het.io (GitHub: hetio/hetionet)

**Why Hetionet:**
- Pre-integrated biomedical knowledge graph with 47,031 nodes and 2,250,197 edges
- Contains 11 node types: Compound, Disease, Gene, Anatomy, Pathway, Side Effect, Pharmacologic Class, Biological Process, Cellular Component, Molecular Function, Symptom
- Contains 24 edge types including Compound-treats-Disease, Compound-targets-Gene, Gene-associates-Disease
- Already used in published drug repurposing research (Himmelstein et al., 2017)
- Clean, well-documented, ready for graph machine learning

**Supplementary Data Sources:**
| Source | Purpose |
|--------|---------|
| DrugBank | Drug mechanism descriptions for RAG |
| CTD | Additional drug-disease-gene associations |
| ClinicalTrials.gov | Validation of repurposing candidates |

### Feature Engineering and Modeling Plan

**Input Representation:**
- Node features: one-hot encoding of node type, plus learned embeddings
- Edge features: relation type encoding
- For compounds: molecular fingerprints (optional enhancement)
- For genes: Gene Ontology annotations
- For diseases: Disease Ontology hierarchy position

**Target Variable:**
- Binary classification: Does edge (Compound, treats, Disease) exist?
- Training uses known treat relationships; negative sampling creates non-treat pairs
- Evaluation on held-out edges tests link prediction capability

**Recommended Architecture: Relational Graph Convolutional Network (R-GCN)**

**Justification:**
- Designed specifically for heterogeneous graphs with multiple relation types
- Learns node embeddings that capture both local structure and global connectivity
- Standard approach for biomedical knowledge graph completion
- Extensible to more sophisticated architectures (CompGCN, HGT) if needed

**Model Flow:**
```
Input Graph (Hetionet)
       │
       ▼
┌─────────────────┐
│  R-GCN Encoder  │  ──▶  Node Embeddings (128-dim)
│  (2-3 layers)   │
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ Score Function  │  ──▶  P(drug treats disease)
│ (DistMult/MLP)  │
└─────────────────┘
```

### Python Environment Setup

**requirements.txt**
```
# Core Data Science
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
scikit-learn>=1.3.0

# Graph Machine Learning
torch>=2.0.0
torch-geometric>=2.4.0
networkx>=3.1

# Knowledge Graph Utilities
pykeen>=1.10.0

# Cheminformatics (optional molecular features)
rdkit>=2023.3.1

# RAG and LLM
chromadb>=0.4.0
sentence-transformers>=2.2.0
openai>=1.0.0
langchain>=0.1.0

# MLOps
mlflow>=2.8.0

# Web Interface
streamlit>=1.28.0
plotly>=5.17.0

# Utilities
pyyaml>=6.0
tqdm>=4.66.0
requests>=2.31.0
python-dotenv>=1.0.0
```

---

## Section 2: RAG and LLM Orchestration Plan

### Knowledge Base Sources

**Source 1: DrugBank Drug Descriptions**
- Mechanism of action text for each compound
- Pharmacodynamics and pharmacokinetics
- Known indications and off-label uses
- Structured extraction from DrugBank XML dump (academic license available)

**Source 2: PubMed Drug Repurposing Literature**
- Query: "(drug repurposing OR drug repositioning) AND {disease_name}"
- Focus on review articles and clinical evidence
- Extract abstracts using Entrez API
- Provides human-validated evidence for candidate explanations

### RAG Pipeline Steps

**Step 1: Knowledge Ingestion**
- Parse DrugBank XML for compound descriptions
- Fetch PubMed abstracts via Entrez API
- Extract pathway descriptions from KEGG/Reactome
- Store raw documents with metadata (source, drug_id, disease_id)

**Step 2: Document Chunking and Embedding**
- Chunk documents by semantic units (mechanism sections, indication paragraphs)
- Generate embeddings using PubMedBERT or BioLinkBERT
- Preserve metadata linkage (chunk to source drug/disease)

**Step 3: Vector Store Indexing**
- Index chunks in ChromaDB with metadata filters
- Enable filtering by drug_id, disease_id, source_type
- Create separate collections for mechanisms vs clinical evidence

**Step 4: Retrieval with Biological Context**
- Given prediction (drug X, disease Y), retrieve:
  - Drug X mechanism documents
  - Disease Y pathophysiology documents
  - Any existing literature mentioning both
  - Pathway information connecting drug targets to disease genes

**Step 5: LLM Synthesis**
- Prompt structure: prediction score + retrieved evidence + reasoning request
- Generate explanation covering: biological plausibility, existing evidence, confidence assessment, suggested validation steps

### Tool Definition for LLM Function Calling

```python
from typing import TypedDict, List, Optional

class RepurposingPrediction(TypedDict):
    drug_id: str
    drug_name: str
    disease_id: str
    disease_name: str
    prediction_score: float
    confidence: str
    shared_genes: List[str]
    shared_pathways: List[str]
    known_similar_indications: List[str]


def get_repurposing_prediction(
    drug_id: str,
    disease_id: str,
    include_pathway_analysis: bool = True,
    top_k_genes: int = 10
) -> RepurposingPrediction:
    """
    Retrieve drug repurposing prediction from the R-GCN model.
    
    Parameters
    ----------
    drug_id : str
        DrugBank identifier (e.g., "DB00945" for Aspirin)
    disease_id : str
        Disease Ontology identifier (e.g., "DOID:1612" for breast cancer)
    include_pathway_analysis : bool
        If True, return shared biological pathways between drug targets
        and disease-associated genes
    top_k_genes : int
        Number of top connecting genes to return in explanation
    
    Returns
    -------
    RepurposingPrediction
        Dictionary containing prediction score (0-1), confidence level,
        list of shared genes connecting drug to disease, shared pathways,
        and known similar indications for context
    
    Raises
    ------
    ValueError
        If drug_id or disease_id not found in knowledge graph
    """
    pass
```

---

## Section 3: Project Structure and Next Steps

### Directory Structure

```
drug_repurposing_system/
│
├── config/
│   ├── model_config.yaml
│   ├── rag_config.yaml
│   └── paths.yaml
│
├── data/
│   ├── raw/
│   │   ├── hetionet/
│   │   ├── drugbank/
│   │   └── pubmed_abstracts/
│   ├── processed/
│   │   ├── graph/
│   │   └── embeddings/
│   └── knowledge_base/
│       └── chunks/
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── hetionet_loader.py
│   │   ├── drugbank_parser.py
│   │   └── graph_builder.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── rgcn_encoder.py
│   │   ├── link_predictor.py
│   │   └── trainer.py
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── document_processor.py
│   │   ├── retriever.py
│   │   └── explanation_generator.py
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predictor.py
│   │   └── tools.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging_config.py
│       └── metrics.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_rag_evaluation.ipynb
│
├── app/
│   ├── streamlit_app.py
│   └── components/
│       ├── search_interface.py
│       ├── prediction_display.py
│       └── graph_visualization.py
│
├── scripts/
│   ├── download_data.py
│   ├── build_knowledge_graph.py
│   ├── train_model.py
│   └── index_documents.py
│
├── mlruns/
│
├── requirements.txt
├── setup.py
├── .env.example
└── README.md
```

### First Coding Task: Hetionet Data Loader

**File:** `src/data/hetionet_loader.py`

**Rationale:** The knowledge graph is the foundation of this project. Loading and understanding the Hetionet structure enables all downstream work: model input preparation, pathway analysis, and RAG context retrieval.

```python
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
from typing import Dict, List, Tuple, Optional
import json
import logging

import torch
from torch_geometric.data import HeteroData
import pandas as pd
import networkx as nx


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
        self.logger = logging.getLogger(__name__)
    
    def load(self) -> HeteroData:
        """
        Load Hetionet and convert to PyTorch Geometric HeteroData.
        
        Returns
        -------
        HeteroData
            Heterogeneous graph with node features and edge indices
        """
        raise NotImplementedError("Implementation in next phase")
    
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
        raise NotImplementedError("Implementation in next phase")
    
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
        raise NotImplementedError("Implementation in next phase")
```

---

## Immediate Next Steps

1. **Create project directory** and initialize git repository
2. **Implement `hetionet_loader.py`** with download and parsing logic
3. **Run exploratory notebook** to understand graph statistics (node counts, edge distributions, connectivity)
4. **Identify target edges** for the link prediction task (Compound-treats-Disease)

Ready to proceed with the implementation of the Hetionet loader?
