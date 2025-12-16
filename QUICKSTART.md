# Drug Repurposing Intelligence System - Quick Start Guide

## Overview

This system uses graph neural networks (R-GCN) and RAG to predict and explain drug repurposing opportunities.

## Setup

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required variables:
- `OPENAI_API_KEY`: For LLM explanations (optional for training)
- `NCBI_API_KEY`: For PubMed access (optional)

## Workflow

### Step 1: Download Data

```bash
python scripts/download_data.py
```

This downloads Hetionet (~170MB uncompressed).

### Step 2: Build Knowledge Graph

```bash
python scripts/build_knowledge_graph.py
```

Processes Hetionet into PyTorch Geometric format with:
- 47K+ nodes across 11 types
- 2.25M+ edges across 24 types
- Node features added automatically

### Step 3: Train Model

```bash
python scripts/train_model.py --epochs 50 --batch-size 512
```

Trains R-GCN for link prediction:
- Predicts Compound-treats-Disease relationships
- Uses negative sampling (5:1 ratio)
- Saves best model to `models/checkpoints/best_model.pt`

Optional arguments:
- `--device cuda`: Use GPU if available
- `--epochs 100`: Train for more epochs
- `--config path/to/config.yaml`: Custom configuration

### Step 4: Index Documents for RAG

```bash
python scripts/index_documents.py
```

Indexes biomedical documents:
- Drug mechanisms of action
- Disease pathophysiology
- Literature evidence

Creates ChromaDB vector database at `data/knowledge_base/chroma/`.

## Usage

### Programmatic Access

```python
from src.inference.predictor import DrugRepurposingPredictor

# Initialize predictor
predictor = DrugRepurposingPredictor(
    model_path="models/checkpoints/best_model.pt"
)

# Make prediction
result = predictor.predict(
    drug_id="DB00945",      # Aspirin
    disease_id="DOID:1612"  # Breast cancer
)

print(f"Score: {result.prediction_score:.3f}")
print(f"Explanation: {result.explanation}")
```

### Using RAG Retriever

```python
from src.rag.retriever import RAGRetriever

retriever = RAGRetriever()

# Retrieve context for a drug
results = retriever.retrieve(
    query="mechanism of action for aspirin",
    collection_name="drug_mechanisms",
    top_k=5
)

for doc in results:
    print(doc['text'])
    print(doc['metadata'])
```

### Generate Explanations

```python
from src.rag.retriever import RAGRetriever
from src.rag.explanation_generator import ExplanationGenerator

retriever = RAGRetriever()
generator = ExplanationGenerator(retriever)

explanation = generator.generate_explanation(
    drug_name="Aspirin",
    disease_name="Breast Cancer",
    prediction_score=0.75
)

print(explanation)
```

## Project Structure

```
Drug_Repurposing_Intelligence_System/
├── config/                     # YAML configuration files
├── data/
│   ├── raw/hetionet/          # Downloaded Hetionet data
│   ├── processed/graph/       # Processed PyG graphs
│   └── knowledge_base/chroma/ # Vector database
├── src/
│   ├── data/                  # Data loading (hetionet_loader.py, graph_builder.py)
│   ├── models/                # R-GCN model (rgcn_encoder.py, link_predictor.py, trainer.py)
│   ├── rag/                   # RAG system (document_processor.py, retriever.py)
│   └── utils/                 # Utilities (logging, metrics)
├── scripts/                   # Executable scripts
└── models/checkpoints/        # Saved model checkpoints
```

## Key Files

- **src/data/hetionet_loader.py**: Loads and parses Hetionet knowledge graph
- **src/data/graph_builder.py**: Converts to PyTorch Geometric format
- **src/models/link_predictor.py**: Drug-disease prediction model
- **src/models/trainer.py**: Training logic with evaluation metrics
- **src/rag/retriever.py**: RAG retrieval for context
- **src/rag/explanation_generator.py**: LLM-based explanations

## Configuration

Edit `config/model_config.yaml`:

```yaml
model:
  encoder:
    hidden_dim: 128      # Embedding dimension
    num_layers: 3        # R-GCN layers
    num_bases: 30        # Basis decomposition

training:
  batch_size: 512
  num_epochs: 100
  learning_rate: 0.001
```

## Evaluation Metrics

The model is evaluated using:
- **Hits@K**: Proportion of correct predictions in top-K
- **MRR**: Mean Reciprocal Rank
- **AUC**: Area under ROC curve

## Next Steps

1. **Improve Features**: Add molecular fingerprints, Gene Ontology annotations
2. **Enhance RAG**: Integrate PubMed literature, clinical trials data
3. **Deploy API**: Create REST API for predictions
4. **Web Interface**: Build Streamlit dashboard (app/streamlit_app.py)
5. **Validation**: Compare predictions with known drug repurposing successes

## Troubleshooting

### CUDA out of memory
- Reduce `batch_size` in config
- Use `--device cpu` for training

### ChromaDB errors
- Ensure `sentence-transformers` is installed
- Check `data/knowledge_base/chroma/` permissions

### Missing Hetionet data
- Run `python scripts/download_data.py` first
- Check internet connection

## References

- **Hetionet**: Himmelstein et al. (2017), eLife
- **R-GCN**: Schlichtkrull et al. (2018), ESWC
- **Drug Repurposing**: Pushpakom et al. (2019), Nature Reviews Drug Discovery

## Support

For issues or questions:
- Check README.md for detailed documentation
- Review configuration files in `config/`
- Examine logs in `logs/` directory
