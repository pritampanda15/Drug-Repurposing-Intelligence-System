# Paper Analysis Feature Guide

## Overview

The Paper Analysis feature allows you to upload scientific publications (PDFs) and automatically:
1. **Extract text** from the PDF
2. **Identify chemical compounds** mentioned in the paper
3. **Visualize 2D molecular structures** of extracted chemicals
4. **Generate AI-powered summaries** of the research

## Installation

Install the required dependencies:

```bash
# Activate your virtual environment first
source .venv/bin/activate  # or: source venv/bin/activate

# Install paper analysis dependencies
pip install pdfplumber PyPDF2 pubchempy httpx certifi
```

## Usage

### 1. Start the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

### 2. Navigate to Paper Analysis Tab

Click on the **"📄 Paper Analysis"** tab in the app.

### 3. Upload a PDF

- Click "Upload PDF Paper"
- Select a scientific paper (PDF format)
- The app will extract text and metadata

### 4. Extract Chemicals

The app automatically identifies potential chemical compounds from the paper text, including:
- Drug names (e.g., Aspirin, Metformin, Imatinib)
- Chemical compounds with common suffixes (-ine, -ide, -ate, -ole, etc.)
- IUPAC-style chemical names

**Features:**
- View list of extracted chemicals with frequency counts
- Select any chemical to visualize its 2D structure
- View molecular properties (MW, LogP, H-bonds, TPSA, etc.)

### 5. Generate Summary

Click **"✨ Generate Summary"** to create an AI-powered analysis including:
- **Main Objective**: Primary research goal
- **Key Methods**: Experimental approaches used
- **Major Findings**: Main results and discoveries
- **Clinical Relevance**: Implications for drug development
- **Key Chemicals**: Important compounds discussed

## Features

### Chemical Extraction
- Pattern-based extraction using regex
- Frequency analysis
- Optional PubChem validation (slower but more accurate)

### Structure Visualization
- 2D molecular structure rendering using RDKit
- Automatic SMILES lookup via PubChem
- Molecular property calculations:
  - Molecular Weight (MW)
  - Lipophilicity (LogP)
  - Hydrogen bond donors/acceptors
  - Rotatable bonds
  - Topological Polar Surface Area (TPSA)

### AI Summarization
- Uses OpenAI GPT models
- Structured output format
- Focuses on pharmaceutical/biomedical relevance

## Requirements

### Required
- Python 3.8+
- pdfplumber or PyPDF2 (PDF extraction)
- RDKit (structure visualization)

### Optional
- OpenAI API key (for AI summaries)
- pubchempy (for chemical validation)

## Configuration

Set your OpenAI API key in `.env`:

```bash
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini  # or gpt-3.5-turbo
```

## Troubleshooting

### "No PDF library available"
Install PDF parsers:
```bash
pip install pdfplumber PyPDF2
```

### "RDKit not installed"
Install RDKit:
```bash
pip install rdkit
```

### "Could not generate structure"
- Chemical name may not be in PubChem database
- Try using the exact IUPAC name
- Some extracted terms may not be actual chemicals

### "Summarization failed"
- Check that `OPENAI_API_KEY` is set in `.env`
- Verify API key is valid
- Check internet connection

## Example Workflow

1. Upload a paper about drug discovery (e.g., a PubMed article PDF)
2. Review extracted chemicals - you'll see drug names and compounds
3. Select "Aspirin" from the list
4. Click "Show 2D Structure" to see the molecular structure
5. View molecular properties
6. Generate summary to get AI analysis of the research

## Supported PDF Formats

- Text-based PDFs (not scanned images)
- Scientific publications
- Drug discovery papers
- Biomedical research articles

## Limitations

- Extraction accuracy depends on PDF quality
- Chemical identification uses pattern matching (may include false positives)
- Structure visualization requires chemicals to be in PubChem database
- AI summaries require valid OpenAI API key
- Very large PDFs (>100 pages) may be slow

## Tips

- For best results, upload papers with clear chemical nomenclature
- The system works best with pharmaceutical/biomedical papers
- Use PubChem validation for more accurate chemical identification (slower)
- Review extracted chemicals before trusting them completely

## Advanced Usage

### Programmatic Access

You can use the modules directly in Python:

```python
from src.paper_analysis.pdf_extractor import PDFExtractor
from src.paper_analysis.chemical_extractor import ChemicalExtractor
from src.paper_analysis.structure_visualizer import StructureVisualizer
from src.paper_analysis.summarizer import PaperSummarizer

# Extract text
extractor = PDFExtractor()
text = extractor.extract_text("paper.pdf")

# Extract chemicals
chem_extractor = ChemicalExtractor()
chemicals = chem_extractor.extract_chemicals(text, validate=True)

# Visualize structure
visualizer = StructureVisualizer()
img_bytes = visualizer.draw_from_name("Aspirin")

# Generate summary
summarizer = PaperSummarizer()
summary = summarizer.summarize_paper(text)
```

## Support

For issues or questions:
- Check the main README.md
- Review error messages in the app
- Ensure all dependencies are installed
- Verify your OpenAI API key is valid

---

Happy analyzing! 🔬📄
