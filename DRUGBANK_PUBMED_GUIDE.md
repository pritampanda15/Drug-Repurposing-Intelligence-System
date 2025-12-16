# DrugBank and PubMed Integration Guide

Complete guide for using DrugBank and PubMed with the Drug Repurposing Intelligence System.

## Overview

The system now supports:
- **DrugBank**: Full database of drug mechanisms, indications, and pharmacology
- **PubMed**: Scientific literature from NCBI's biomedical database

## Setup

### 1. Install Dependencies

```bash
pip install biopython
```

Or reinstall all requirements:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Edit your `.env` file:

```bash
# Required for PubMed
NCBI_EMAIL=your_email@example.com

# Optional but recommended (increases rate limit from 3 to 10 req/sec)
NCBI_API_KEY=your_ncbi_api_key_here

# Optional for LLM explanations
OPENAI_API_KEY=your_openai_key
```

**Get NCBI API Key:**
1. Create account at https://www.ncbi.nlm.nih.gov/account/
2. Go to Settings → API Key Management
3. Create a new API key
4. Add to `.env` file

## DrugBank Setup

### 1. Download DrugBank

1. Visit https://go.drugbank.com/releases/latest
2. Create a free academic account
3. Download **"Full Database"** in XML format
4. Save as `data/raw/drugbank/full_database.xml`

```bash
mkdir -p data/raw/drugbank
# Move downloaded file here
mv ~/Downloads/full\ database.xml data/raw/drugbank/full_database.xml
```

### 2. Index DrugBank

```bash
# Index DrugBank drugs (this will take 5-10 minutes)
python3 scripts/index_documents.py --drugbank data/raw/drugbank/full_database.xml
```

This will:
- Parse ~13,000+ drug entries from DrugBank XML
- Extract mechanisms of action, indications, pharmacodynamics
- Create embeddings using BioLinkBERT
- Store in ChromaDB vector database

**What gets indexed:**
- Drug mechanisms of action
- Indications (what diseases the drug treats)
- Pharmacodynamics (how the drug works)
- Drug categories and classifications
- Only small molecule drugs (excludes biologics)

## PubMed Integration

### 1. Index PubMed Literature

```bash
# Index general drug repurposing literature
python3 scripts/index_documents.py --pubmed

# Custom queries
python3 scripts/index_documents.py --pubmed \
  --pubmed-queries "drug repurposing cancer" "drug repositioning diabetes"
```

**Options:**
- `--pubmed`: Enable PubMed indexing
- `--pubmed-queries`: Custom search queries (default: "drug repurposing", "drug repositioning")
- Each query fetches up to 50 most relevant articles

**What gets indexed:**
- Article titles
- Abstracts
- Authors, journal, publication year
- PubMed ID (PMID) for reference

### 2. Rate Limits

- **Without API key**: 3 requests/second
- **With API key**: 10 requests/second

For large-scale indexing, get an API key to speed up the process.

## Complete Indexing Workflow

### Full Pipeline

```bash
# 1. Index everything: DrugBank + PubMed + sample data
python3 scripts/index_documents.py \
  --drugbank data/raw/drugbank/full_database.xml \
  --pubmed \
  --pubmed-queries "drug repurposing" "drug repositioning" "drug targets"

# 2. Or skip sample data and use only real data
python3 scripts/index_documents.py \
  --skip-sample \
  --drugbank data/raw/drugbank/full_database.xml \
  --pubmed
```

### Expected Output

```
============================================================
Indexing Documents for RAG System
============================================================

Indexing sample biomedical data...
✓ Indexed 5 drug mechanism documents

============================================================
Processing DrugBank
============================================================
Parsing DrugBank XML (this may take a few minutes)...
Parsing DrugBank: 13562it [02:34, 87.63it/s]
Parsed 13562 drugs from DrugBank
Created 24789 documents from 13562 drugs
Indexed 100/24789 drugs
...
✓ DrugBank indexing complete: 13562 drugs, 24789 documents

============================================================
Processing PubMed Literature
============================================================
Fetching articles for: drug repurposing
Found 50 articles
Fetched 50 abstracts
  Added 50 articles
✓ Indexed 50 PubMed articles

============================================================
✓ Document indexing complete!
```

## Using the Indexed Data

### 1. Query RAG System

```python
from src.rag.retriever import RAGRetriever

retriever = RAGRetriever()

# Search drug mechanisms
results = retriever.retrieve(
    query="aspirin mechanism of action",
    collection_name="drug_mechanisms",
    top_k=5
)

for doc in results:
    print(doc['text'])
    print(doc['metadata'])
```

### 2. Generate Explanations

```python
from src.inference.predictor import DrugRepurposingPredictor

predictor = DrugRepurposingPredictor(
    model_path="models/checkpoints/best_model.pt",
    enable_rag=True  # Use RAG for explanations
)

result = predictor.predict(
    drug_name="Aspirin",
    disease_name="Breast Cancer",
    generate_explanation=True
)

print(f"Score: {result.prediction_score:.3f}")
print(f"\nExplanation:\n{result.explanation}")
```

### 3. Streamlit Interface

```bash
streamlit run app/streamlit_app.py
```

The web interface will now use:
- All indexed DrugBank drugs (instead of just samples)
- PubMed literature for context
- Full RAG-powered explanations

## Advanced Usage

### Custom PubMed Queries

Index specific drug-disease combinations:

```python
from src.rag.pubmed_fetcher import PubMedFetcher
from src.rag.document_processor import DocumentProcessor

fetcher = PubMedFetcher()
processor = DocumentProcessor()

# Search for specific drug-disease pair
pmids = fetcher.search_drug_disease(
    drug_name="Metformin",
    disease_name="Cancer",
    max_results=20
)

articles = fetcher.fetch_abstracts(pmids)
texts, metadatas, ids = fetcher.create_documents(
    articles,
    drug_name="Metformin",
    disease_name="Cancer"
)

collection = processor.create_collection("pubmed_literature")
processor.add_documents(collection, texts, metadatas, ids)
```

### Filtering Searches

```python
# Search with metadata filters
results = retriever.retrieve(
    query="cancer treatment",
    collection_name="drug_mechanisms",
    top_k=10,
    filters={"groups": "approved"}  # Only approved drugs
)

# Search PubMed by year
results = retriever.retrieve(
    query="drug repurposing",
    collection_name="pubmed_literature",
    filters={"year": "2023"}
)
```

## Data Statistics

After full indexing, you should have:

| Source | Documents | Content |
|--------|-----------|---------|
| DrugBank | ~25,000 | Drug mechanisms and indications |
| PubMed | 50-500+ | Scientific literature abstracts |
| Sample Data | 5 | Example drugs (optional) |

## Troubleshooting

### DrugBank Issues

**Error: "File does not exist"**
```bash
# Check file location
ls -lh data/raw/drugbank/full_database.xml

# If missing, download from DrugBank website
```

**Error: "XML parsing failed"**
- Ensure you downloaded the XML format (not RDF or other formats)
- File should be ~1GB uncompressed
- Check for corruption: `file data/raw/drugbank/full_database.xml`

### PubMed Issues

**Error: "Email required for NCBI"**
```bash
# Add to .env file
echo "NCBI_EMAIL=your@email.com" >> .env
```

**Error: "BioPython not installed"**
```bash
pip install biopython
```

**Rate Limiting**
- If getting errors, add `NCBI_API_KEY` to increase rate limit
- Or reduce `max_results` in queries

### Memory Issues

If parsing DrugBank causes memory issues:

```python
# The parser uses iterative parsing to minimize memory
# But for very limited memory, process in batches:

from src.data.drugbank_parser import DrugBankParser

parser = DrugBankParser("data/raw/drugbank/full_database.xml")

# Parse in smaller chunks
# (modify parser to add batch support if needed)
```

## Performance Tips

1. **Use API Key**: Get NCBI API key for faster PubMed indexing
2. **Cache Embeddings**: Once indexed, embeddings are cached in ChromaDB
3. **Batch Processing**: DrugBank indexing happens in batches of 100
4. **GPU Acceleration**: Use CUDA-capable GPU for faster embedding generation

## Next Steps

After indexing:

1. **Train Model**: `python3 scripts/train_model.py --epochs 50`
2. **Make Predictions**: Use predictor with full drug database
3. **Launch Web App**: `streamlit run app/streamlit_app.py`
4. **Explore Data**: Query ChromaDB collections

## References

- DrugBank: https://go.drugbank.com/
- PubMed: https://pubmed.ncbi.nlm.nih.gov/
- NCBI E-utilities: https://www.ncbi.nlm.nih.gov/books/NBK25501/
- BioPython: https://biopython.org/
