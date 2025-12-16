# Drug Repurposing Intelligence System

An AI-powered system for discovering novel drug-disease relationships using graph neural networks and retrieval-augmented generation (RAG).

## Overview

This project combines:
- **Relational Graph Convolutional Networks (R-GCN)** for predicting drug-disease relationships from biomedical knowledge graphs
- **RAG with LLM** for generating human-interpretable explanations of predictions
- **Hetionet** as the primary knowledge graph (47,031 nodes, 2.25M edges)

## Key Features

- **Graph Neural Network Link Prediction**: Predicts which drugs may treat which diseases using R-GCN on heterogeneous biomedical graphs
- **Explainable AI**: Generates natural language explanations citing mechanisms of action, biological pathways, and literature evidence
- **Multi-Source Knowledge Integration**: Combines DrugBank, PubMed, KEGG, and Reactome data
- **Interactive Web Interface**: Streamlit-based UI for exploring predictions and explanations

## Architecture

```
Input: Drug + Disease Query
         │
         ▼
┌─────────────────────┐
│   R-GCN Model       │  → Prediction Score (0-1)
│  (Link Prediction)  │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   RAG Pipeline      │  → Biological Context
│  (ChromaDB + LLM)   │    - Mechanisms
└─────────────────────┘    - Pathways
         │                 - Literature Evidence
         ▼
    Explanation + Confidence Assessment
```

## Project Structure

```
drug_repurposing_system/
├── config/                  # Configuration files
│   ├── model_config.yaml   # R-GCN model parameters
│   ├── rag_config.yaml     # RAG pipeline configuration
│   └── paths.yaml          # Data and model paths
│
├── data/                   # Data storage
│   ├── raw/                # Original datasets
│   ├── processed/          # Processed graphs and features
│   └── knowledge_base/     # RAG document store
│
├── src/                    # Source code
│   ├── data/               # Data loading and processing
│   ├── models/             # R-GCN implementation
│   ├── rag/                # RAG pipeline
│   ├── inference/          # Prediction and tools
│   └── utils/              # Utilities
│
├── notebooks/              # Jupyter notebooks for analysis
├── app/                    # Streamlit web interface
└── scripts/                # Executable scripts
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd Drug_Repurposing_Intelligence_System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install project in editable mode
pip install -e .
```

### Environment Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Required environment variables:
- `OPENAI_API_KEY`: For LLM-based explanations
- `NCBI_API_KEY`: For PubMed literature retrieval
- `NCBI_EMAIL`: Your email for NCBI API access

## Quick Start

### 1. Download Data

```bash
python scripts/download_data.py
```

This downloads:
- Hetionet knowledge graph
- DrugBank data (requires academic license)

### 2. Build Knowledge Graph

```bash
python scripts/build_knowledge_graph.py
```

Converts Hetionet to PyTorch Geometric format.

### 3. Train Model

```bash
python scripts/train_model.py
```

Trains R-GCN model for link prediction. Monitor with MLflow:

```bash
mlflow ui
```

### 4. Index Documents for RAG

```bash
python scripts/index_documents.py
```

Processes and indexes documents in ChromaDB.

### 5. Launch Web Interface

```bash
streamlit run app/streamlit_app.py
```

## Usage

### Programmatic Access

```python
from src.inference.predictor import DrugRepurposingPredictor

# Initialize predictor
predictor = DrugRepurposingPredictor(
    model_path="models/checkpoints/best_model.pt",
    config_path="config/model_config.yaml"
)

# Get prediction
result = predictor.predict(
    drug_id="DB00945",      # Aspirin
    disease_id="DOID:1612"  # Breast cancer
)

print(f"Prediction Score: {result.prediction_score:.3f}")
print(f"Confidence: {result.confidence}")
print(f"\nExplanation:\n{result.explanation}")
```

### LLM Tool Calling

The system provides a tool for LLM agents:

```python
from src.inference.tools import get_repurposing_prediction

prediction = get_repurposing_prediction(
    drug_id="DB00945",
    disease_id="DOID:1612",
    include_pathway_analysis=True,
    top_k_genes=10
)
```

## Data Sources

| Source | Purpose | License |
|--------|---------|---------|
| [Hetionet](https://het.io) | Primary knowledge graph | CC0 1.0 |
| [DrugBank](https://go.drugbank.com) | Drug mechanisms | Academic license required |
| [PubMed](https://pubmed.ncbi.nlm.nih.gov) | Literature evidence | Public domain |
| [KEGG](https://www.kegg.jp) | Pathway information | Academic use |
| [Reactome](https://reactome.org) | Pathway information | CC BY 4.0 |

## Model Details

### R-GCN Architecture

- **Input**: Hetionet heterogeneous graph
- **Encoder**: 3-layer R-GCN with basis decomposition
- **Decoder**: DistMult scoring function
- **Output**: Drug-disease relationship probability

### Training

- **Task**: Link prediction (binary classification)
- **Positive samples**: Known Compound-treats-Disease edges
- **Negative sampling**: 5:1 ratio
- **Evaluation**: Hits@K, MRR, AUC-ROC

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
# Format code
black src/

# Lint
flake8 src/
```

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Citation

If you use this system in your research, please cite:

```bibtex
@software{drug_repurposing_system,
  title={Drug Repurposing Intelligence System},
  author={},
  year={2025},
  url={https://github.com/...}
}
```

### Key References

- Himmelstein et al. (2017). "Systematic integration of biomedical knowledge prioritizes drugs for repurposing." *eLife*.
- Schlichtkrull et al. (2018). "Modeling Relational Data with Graph Convolutional Networks." *ESWC*.

## License

MIT License - See LICENSE file for details

## Support

For questions or issues:
- Open a GitHub issue
- Contact: [your-email]

## Roadmap

- [ ] Implement full Hetionet loader
- [ ] Train baseline R-GCN model
- [ ] Build RAG pipeline
- [ ] Add molecular fingerprint features
- [ ] Integrate clinical trial data
- [ ] Deploy as web service
- [ ] Add batch prediction API
- [ ] Publish validation study
