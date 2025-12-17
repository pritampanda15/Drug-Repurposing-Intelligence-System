# Drug Repurposing Intelligence System - Complete Setup Guide

Step-by-step guide from scratch to a fully functional system.

## 📋 Prerequisites

- Python 3.8 or higher
- 8GB+ RAM (16GB recommended for training)
- ~2GB free disk space for data
- (Optional) CUDA-capable GPU for faster training
- (Optional) OpenAI API key for AI explanations

## 🚀 Step-by-Step Setup

### Step 1: Clone and Setup Environment

```bash
# Navigate to project directory
cd Drug_Repurposing_Intelligence_System

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install project in editable mode
pip install -e .
```

**Expected time**: 5-10 minutes

### Step 2: Configure Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env file with your settings
nano .env  # or use any text editor
```

**Required settings** in `.env`:
```bash
# Optional but recommended for AI explanations
OPENAI_API_KEY=your_openai_api_key_here

# Optional for PubMed literature
NCBI_EMAIL=your_email@example.com
NCBI_API_KEY=your_ncbi_api_key  # Optional, increases rate limit
```

**How to get API keys:**
- OpenAI: https://platform.openai.com/api-keys
- NCBI: https://www.ncbi.nlm.nih.gov/account/settings/

### Step 3: Download Hetionet Data

```bash
# Download Hetionet knowledge graph (~170MB)
python3 scripts/download_data.py
```

**What this does:**
- Downloads Hetionet v1.0 from GitHub
- Decompresses the file automatically
- Creates README files with documentation

**Expected output:**
```
============================================================
Drug Repurposing Intelligence System - Data Download
============================================================
Downloading Hetionet knowledge graph...
Source: https://het.io
Nodes: 47,031 | Edges: 2,250,197
hetionet-v1.0.json: 100%|██████████| 171M/171M [00:45<00:00, 3.8MB/s]
Decompressing hetionet-v1.0.json.bz2
✓ Hetionet download complete
```

**Expected time**: 1-2 minutes (depends on internet speed)

### Step 4: Build Knowledge Graph

```bash
# Process Hetionet into PyTorch Geometric format
python3 scripts/build_knowledge_graph.py
```

**What this does:**
- Parses 47,031 nodes and 2,250,197 edges
- Converts to PyTorch Geometric HeteroData
- Adds node features (128-dimensional embeddings)
- Adds reverse edges for undirected graph
- Caches processed graph for faster loading

**Expected output:**
```
============================================================
Building Knowledge Graph for Drug Repurposing
============================================================
Loading Hetionet from data/raw/hetionet/hetionet-v1.0.json
Loaded 47031 nodes and 2250197 edges
Parsing nodes: 100%|██████████| 47031/47031
Parsing edges: 100%|██████████| 2250197/2250197
Caching processed data...

GRAPH STATISTICS
============================================================
Node Types: 11
  Compound: 1538 nodes, 128D features
  Disease: 137 nodes, 128D features
  Gene: 20945 nodes, 128D features
  ...

Edge Types: 48 (24 original + 24 reverse)
Total Nodes: 47,031
Total Edges: 4,500,394
============================================================

✓ Knowledge graph built successfully!
Saved to: data/processed/graph/hetionet_graph.pt
```

**Expected time**: 2-5 minutes

### Step 5: Train the Model

```bash
# Train R-GCN model (basic training)
python3 scripts/train_model.py --epochs 20 --batch-size 512
```

**For better results** (takes longer):
```bash
# Extended training with GPU
python3 scripts/train_model.py --epochs 100 --batch-size 1024 --device cuda
```

**What this does:**
- Loads knowledge graph
- Extracts Compound-treats-Disease relationships
- Generates negative samples (5:1 ratio)
- Trains R-GCN model with link prediction
- Saves best model checkpoint
- Tracks training with MLflow

**Expected output:**
```
============================================================
Training Drug Repurposing Model
============================================================
Loading Hetionet data...
Loaded graph with 47031 nodes and 2250197 edges
Preparing training data...
Positive pairs: 755
Negative pairs: 3775
Train: 3171, Val: 680, Test: 679
Model initialized with 278,016 parameters

Epoch 1/20
Training: 100%|██████████| 7/7 [00:02<00:00]
Train Loss: 0.6847, Val Loss: 0.6753
✓ New best model saved (val_loss: 0.6753)

Epoch 2/20
...

Evaluating on test set...
Test Loss: 0.6421

============================================================
✓ Training complete!
Best model saved to: models/checkpoints/best_model.pt
============================================================
```

**Expected time**: 5-30 minutes (depending on epochs and hardware)

### Step 6: Index Documents for RAG (Optional)

```bash
# Index sample data only (quick)
python3 scripts/index_documents.py

# Or index with DrugBank (requires download)
python3 scripts/index_documents.py \
  --drugbank data/raw/drugbank/full_database.xml

# Or index everything including PubMed
python3 scripts/index_documents.py \
  --drugbank data/raw/drugbank/full_database.xml \
  --pubmed \
  --skip-sample
```

**Prerequisites for full indexing:**
1. **DrugBank**: Download from https://go.drugbank.com/releases/latest
   - Create free academic account
   - Download "Full Database" XML
   - Save to `data/raw/drugbank/full_database.xml`

2. **PubMed**: Set `NCBI_EMAIL` in `.env` file

**Expected output:**
```
============================================================
Indexing Documents for RAG System
============================================================
Indexing sample biomedical data...
Creating collection 'drug_mechanisms'
Generating embeddings for 5 documents...
✓ Indexed 5 drug mechanism documents

============================================================
✓ Document indexing complete!
ChromaDB location: data/knowledge_base/chroma
```

**Expected time**: 1 minute (sample) to 10+ minutes (full DrugBank)

### Step 7: Launch Web Interface

```bash
# Start Streamlit app
streamlit run app/streamlit_app.py
```

**Expected output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.x:8501
```

Open your browser to `http://localhost:8501`

**Expected time**: 5-10 seconds to start

## ✅ Verification Checklist

After setup, verify everything works:

- [ ] Hetionet data downloaded (check `data/raw/hetionet/hetionet-v1.0.json` exists)
- [ ] Knowledge graph built (check `data/processed/graph/hetionet_graph.pt` exists)
- [ ] Model trained (check `models/checkpoints/best_model.pt` exists)
- [ ] Streamlit app loads without errors
- [ ] Can select drugs and diseases from dropdowns
- [ ] Prediction button generates scores
- [ ] (Optional) RAG explanations work if configured

## 🎯 Quick Test

Test the system with a simple prediction:

```python
from src.inference.predictor import DrugRepurposingPredictor

# Initialize predictor
predictor = DrugRepurposingPredictor(
    model_path="models/checkpoints/best_model.pt"
)

# Make a prediction
result = predictor.predict(
    drug_name="Aspirin",
    disease_name="rheumatoid arthritis"
)

print(f"Drug: {result.drug_name}")
print(f"Disease: {result.disease_name}")
print(f"Prediction Score: {result.prediction_score:.3f}")
print(f"Confidence: {result.confidence}")
```

## 📊 What You Can Do Now

### 1. Web Interface (Streamlit)
```bash
streamlit run app/streamlit_app.py
```
- Select any drug and disease
- Get repurposing predictions
- View graph statistics
- Explore knowledge graph

### 2. Programmatic Access
```python
from src.inference.predictor import DrugRepurposingPredictor

predictor = DrugRepurposingPredictor()

# Single prediction
result = predictor.predict(
    drug_name="Metformin",
    disease_name="Cancer"
)

# Find top diseases for a drug
results = predictor.predict_top_k(
    drug_name="Aspirin",
    top_k=10
)
```

### 3. Query Knowledge Base
```python
from src.rag.retriever import RAGRetriever

retriever = RAGRetriever()

results = retriever.retrieve(
    query="aspirin mechanism of action",
    collection_name="drug_mechanisms",
    top_k=5
)
```

### 4. Monitor Training
```bash
# View MLflow experiments
mlflow ui

# Open browser to http://localhost:5000
```

## 🐛 Troubleshooting

### Issue: "Model not found"
**Solution**: Train the model first with `python3 scripts/train_model.py`

### Issue: "CUDA out of memory"
**Solution**: 
```bash
# Use CPU instead
python3 scripts/train_model.py --device cpu

# Or reduce batch size
python3 scripts/train_model.py --batch-size 256 --device cuda
```

### Issue: "RAG explanations don't work"
**Solution**:
1. Index documents: `python3 scripts/index_documents.py`
2. Set `OPENAI_API_KEY` in `.env`
3. Restart Streamlit

### Issue: "Streamlit app shows no drugs/diseases"
**Solution**: Run `python3 scripts/build_knowledge_graph.py` first

### Issue: "Import errors"
**Solution**:
```bash
# Reinstall dependencies
pip install -r requirements.txt
pip install -e .
```

## 📁 Directory Structure After Setup

```
Drug_Repurposing_Intelligence_System/
├── data/
│   ├── raw/
│   │   └── hetionet/
│   │       ├── hetionet-v1.0.json          ✓ Downloaded
│   │       └── hetionet_processed.pkl       ✓ Cached
│   ├── processed/
│   │   └── graph/
│   │       └── hetionet_graph.pt            ✓ Built
│   └── knowledge_base/
│       └── chroma/                          ✓ Indexed (optional)
├── models/
│   └── checkpoints/
│       └── best_model.pt                    ✓ Trained
├── logs/
│   └── training/
│       └── train.log                        ✓ Generated
└── mlruns/                                  ✓ MLflow tracking
```

## 🚀 Next Steps

1. **Improve Model**: Train for more epochs (50-100)
2. **Add Data**: Index DrugBank and PubMed
3. **Explore**: Use Streamlit to find novel drug-disease pairs
4. **Validate**: Compare predictions with known repurposing successes
5. **Extend**: Add your own features or data sources

## 📚 Additional Resources

- **QUICKSTART.md**: Quick reference guide
- **DRUGBANK_PUBMED_GUIDE.md**: DrugBank and PubMed integration
- **README.md**: Full documentation
- **instructions.md**: Original planning document

## 🔗 Useful Links

- Hetionet: https://het.io
- DrugBank: https://go.drugbank.com
- PubMed: https://pubmed.ncbi.nlm.nih.gov
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io
- Streamlit: https://docs.streamlit.io

## 💡 Tips

- Use `--help` with any script to see all options
- Check logs in `logs/` if something fails
- Use GPU for faster training if available
- Start with small epochs for testing
- RAG is optional - predictions work without it
