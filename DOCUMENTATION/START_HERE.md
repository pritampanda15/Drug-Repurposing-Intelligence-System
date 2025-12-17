# 🚀 START HERE - Drug Repurposing Intelligence System

## Quick Start: 7 Simple Steps

Follow these steps in order to get your system running:

### ✅ Step 1: Setup Environment (5 minutes)

```bash
# Activate virtual environment
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies (if not already done)
pip install -r requirements.txt
pip install -e .
```

### ✅ Step 2: Configure API Keys (2 minutes)

Edit `.env` file:
```bash
# Required for PubMed (optional)
NCBI_EMAIL=your_email@example.com

# Required for AI explanations (optional)
OPENAI_API_KEY=sk-your-key-here
```

**Note**: Both are optional. The system works without them - you just won't get AI explanations.

### ✅ Step 3: Download Data (2 minutes)

```bash
python3 scripts/download_data.py
```

Downloads Hetionet (~170MB) with 47K nodes and 2.25M edges.

### ✅ Step 4: Build Knowledge Graph (3 minutes)

```bash
python3 scripts/build_knowledge_graph.py
```

Converts Hetionet to PyTorch Geometric format.

### ✅ Step 5: Train Model (5-30 minutes)

```bash
# Quick training (5 minutes)
python3 scripts/train_model.py --epochs 20

# Better results (20-30 minutes)
python3 scripts/train_model.py --epochs 100
```

Trains R-GCN for drug-disease prediction.

### ✅ Step 6: Index Documents - OPTIONAL (1-10 minutes)

```bash
# Quick: Sample data only
python3 scripts/index_documents.py

# Full: DrugBank + PubMed (if you have them)
python3 scripts/index_documents.py \
  --drugbank data/raw/drugbank/full_database.xml \
  --pubmed
```

**Skip this if**: You don't have DrugBank or want to test first.

### ✅ Step 7: Launch App (10 seconds)

```bash
streamlit run app/streamlit_app.py
```

Opens in browser at http://localhost:8501

## 🎯 What You Get

After these steps, you can:

### 1. **Web Interface** (Streamlit)
- Select from **1,538 drugs** and **137 diseases**
- Get AI-powered repurposing predictions
- View confidence scores and recommendations
- (Optional) AI explanations if configured

### 2. **Programmatic Access**
```python
from src.inference.predictor import DrugRepurposingPredictor

predictor = DrugRepurposingPredictor()
result = predictor.predict(
    drug_name="Aspirin",
    disease_name="Cancer"
)
print(f"Score: {result.prediction_score:.3f}")
```

### 3. **Top-K Predictions**
```python
# Find best diseases for a drug
results = predictor.predict_top_k(
    drug_name="Metformin",
    top_k=10
)
for r in results:
    print(f"{r.disease_name}: {r.prediction_score:.3f}")
```

## 📊 System Overview

```
1. Hetionet Data
   ↓ (download_data.py)
2. Knowledge Graph
   ↓ (build_knowledge_graph.py)
3. Trained Model
   ↓ (train_model.py)
4. Predictions
   → Streamlit App or Python API
```

## 🆘 Troubleshooting

### "Model not found"
→ Run Step 5 (train_model.py)

### "Graph not found"
→ Run Step 4 (build_knowledge_graph.py)

### "Hetionet data not found"
→ Run Step 3 (download_data.py)

### "No explanations generated"
→ This is normal if you skipped Step 6 or don't have OpenAI API key
→ Predictions still work without explanations!

### "CUDA out of memory"
→ Add `--device cpu` to train_model.py command

## 📚 Full Documentation

- **SETUP_GUIDE.md** - Detailed step-by-step guide
- **QUICKSTART.md** - Quick reference
- **DRUGBANK_PUBMED_GUIDE.md** - DrugBank and PubMed integration
- **README.md** - Complete documentation

## ⏱️ Time Requirements

**Minimum** (testing):
- Setup: 5 min
- Download: 2 min
- Build graph: 3 min
- Train (20 epochs): 5 min
- **Total: ~15 minutes**

**Recommended** (better results):
- Setup: 5 min
- Download: 2 min
- Build graph: 3 min
- Train (100 epochs): 20-30 min
- Index docs: 10 min
- **Total: ~45-60 minutes**

## 🎓 Example Session

```bash
# 1. Setup (one time)
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 2. Prepare data
python3 scripts/download_data.py
python3 scripts/build_knowledge_graph.py

# 3. Train model
python3 scripts/train_model.py --epochs 50

# 4. (Optional) Add RAG
python3 scripts/index_documents.py

# 5. Launch app
streamlit run app/streamlit_app.py

# That's it! 🎉
```

## 🔥 Quick Test

Once setup is complete, test it:

```python
python3 << 'EOF'
from src.inference.predictor import DrugRepurposingPredictor

predictor = DrugRepurposingPredictor()

result = predictor.predict(
    drug_name="Aspirin",
    disease_name="rheumatoid arthritis"
)

print(f"\nPrediction for {result.drug_name} → {result.disease_name}")
print(f"Score: {result.prediction_score:.3f}")
print(f"Confidence: {result.confidence}")
print("✓ System working!")
EOF
```

## 💡 Tips

1. **Start simple**: Run with 20 epochs first to test
2. **GPU helps**: Use `--device cuda` if you have a GPU
3. **RAG is optional**: System works without explanations
4. **Monitor training**: Run `mlflow ui` in another terminal
5. **Save time**: Processed data is cached automatically

## ✨ You're Ready!

The system is production-ready and fully functional. Start with the steps above and you'll have a working drug repurposing prediction system in ~15-60 minutes.

For questions, check the documentation files or the detailed SETUP_GUIDE.md.

Happy drug hunting! 💊🔬
