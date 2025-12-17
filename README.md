# Drug Repurposing Intelligence System

An AI-powered system for discovering novel drug-disease relationships using **Relational Graph Convolutional Networks (R-GCN)** and **Retrieval-Augmented Generation (RAG)**. This project integrates heterogeneous biomedical knowledge graphs with natural language processing to predict and explain potential drug repurposing candidates.

![alt image](https://github.com/pritampanda15/Drug-Repurposing-Intelligence-System/blob/main/Intelligence.png)
---

## 🌟 Overview

The system addresses the high cost and time of traditional drug discovery by leveraging existing biomedical data to find "new tricks for old drugs." It predicts which drugs may treat specific diseases and provides human-interpretable explanations by citing mechanisms of action and literature.

### Core Technologies

* **Graph Machine Learning**: 3-layer R-GCN encoder and DistMult decoder on **Hetionet** (47,031 nodes, 2,250,197 edges).
* **Explainable AI (XAI)**: RAG with LLMs to generate natural language explanations citing mechanisms of action and literature.
* **Multi-Source Integration**: Data from Hetionet, DrugBank, PubMed, KEGG, and Reactome.
* **Scientific Paper Analysis**: Specialized module for extracting chemical entities and summarizing PDF publications.

---

## 🏗️ System Architecture & Workflow

The system operates as a modular pipeline that connects deep learning on graphs with semantic search and large language models.

### The Two-Stage Discovery Pipeline

1. **Link Prediction (Deep Learning)**:
* The **R-GCN model** acts as the discovery engine. It analyzes the topology of the Hetionet graph (including gene associations and molecular functions).
* It calculates the probability of a "treats" relationship between a **Compound** and a **Disease** node.


2. **RAG Pipeline (Explanability)**:
* Once a high-probability prediction is made, the system queries a **ChromaDB** vector database containing DrugBank pharmacology and PubMed abstracts.
* An LLM synthesizes this evidence to provide a biological rationale.



### Workflow Diagram

```text
Input: Drug + Disease Query
         │
         ▼
┌──────────────────────────────┐
│       R-GCN Model            │
│ (Heterogeneous Link Prediction) ──▶  Prediction Score (0.0 - 1.0)
│    - 3-Layer R-GCN Encoder   │       Confidence Assessment
│    - DistMult Scoring Decoder│
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│        RAG Pipeline          │
│  (ChromaDB + OpenAI/LLM)     │
├──────────────────────────────┤
│ 1. Vector Search (DrugBank)  │──▶  Biological Context:
│ 2. Lit. Search (PubMed)      │     - Mechanisms of Action
│ 3. Pathway Mapping (KEGG)    │     - Pathway Intersections
└──────────────┬───────────────┘     - Literature Citations
               │
               ▼
      Human-Interpretable 
     Discovery Explanation

```

---

## 🔬 Paper Analysis Algorithm

The "Paper Analysis" feature is a specialized sub-system designed to ingest unstructured research papers and output structured chemical intelligence.

### 1. Document Parsing & NLP

* **Extraction**: Uses `pdfplumber` and `PyPDF2` to extract text from multi-column scientific layouts.
* **Entity Recognition**: Implements a regex-based **Chemical Entity Recognition (CER)** algorithm to find drug names and IUPAC nomenclature (e.g., words ending in `-ine`, `-ide`, `-ate`).
* **Frequency Mapping**: Scores importance based on mention frequency throughout the document.

### 2. Molecular Informatics Workflow

For every extracted chemical, the system runs the following:

* **Identity Validation**: Queries the **PubChem API** via `pubchempy` to verify the entity and fetch its **SMILES** string.
* **2D Rendering**: Uses **RDKit** to generate high-resolution molecular structure diagrams.
* **Descriptor Calculation**: Computes ADME/Tox-relevant properties:
* **Molecular Weight (MW)**
* **Lipophilicity (LogP)**
* **Polar Surface Area (TPSA)**
* **H-Bond Donors/Acceptors**



### 3. AI Summarization Logic

* **Contextual Slicing**: The system extracts the first and last 4,000 characters (Abstract/Introduction and Conclusion/Results) to avoid token limit issues while maintaining context.
* **Structured Prompting**: The LLM is instructed to return a JSON-structured summary covering:
* **Objective**: The core research question.
* **Methods**: Experimental or computational approaches used.
* **Key Findings**: Major results and statistical significance.
* **Clinical Relevance**: Potential for drug repurposing or therapeutic use.



---

## 🛠️ Installation & Execution

### Setup

```bash
# Environment Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Configure API Keys in .env
echo "OPENAI_API_KEY=your_key" > .env
echo "NCBI_EMAIL=your_email@example.com" >> .env

```

### Full Workflow Execution: Refer [SETUP GUIDE](https://github.com/pritampanda15/Drug-Repurposing-Intelligence-System/blob/main/DOCUMENTATION/SETUP_GUIDE.md)

1. **Prepare Data**: `python3 scripts/download_data.py && python3 scripts/build_knowledge_graph.py`
2. **Train Discovery Model**: `python3 scripts/train_model.py --epochs 50 --device cuda`
3. **Index Knowledge Base**: `python3 scripts/index_documents.py --drugbank path/to/db.xml --pubmed`
4. **Run Dashboard**: `streamlit run app/streamlit_app.py`

---

## 📂 Project Structure

* `app/`: Streamlit UI and "Paper Analysis" dashboard.
* `src/models/`: R-GCN architecture and DistMult scoring implementation.
* `src/paper_analysis/`: Modules for PDF parsing, CER, and RDKit visualization.
* `src/rag/`: Vector database management and LLM prompt engineering.
* `data/`: Stores Hetionet JSON, processed `.pt` graph files, and ChromaDB indices.

---

## 📚 References & Data Sources

* **Hetionet**: Himmelstein, et al. (2017) *eLife*.
* **R-GCN**: Schlichtkrull, et al. (2018) *ESWC*.
* **DrugBank**: Wishart DS, et al. *Nucleic Acids Res*.
* **PubMed**: National Center for Biotechnology Information (NCBI).
