#!/usr/bin/env python3
"""
Test script for paper analysis modules.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("Testing Paper Analysis Modules")
print("=" * 70)

# Test 1: Chemical Extractor
print("\n1. Testing Chemical Extractor...")
try:
    from src.paper_analysis.chemical_extractor import ChemicalExtractor

    sample_text = """
    In this study, we investigated the effects of Aspirin and Metformin on cancer cells.
    The compounds were tested at concentrations of 10 μM. We also evaluated Imatinib
    and other tyrosine kinase inhibitors including Gefitinib and Erlotinib.
    The chemical formula H2O was used as a control.
    """

    extractor = ChemicalExtractor()
    chemicals = extractor.extract_chemicals(sample_text, min_length=4, max_results=10)

    print(f"   ✓ Found {len(chemicals)} chemicals")
    for chem in chemicals[:5]:
        print(f"     - {chem['name']} (mentioned {chem['count']} times)")

except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 2: Structure Visualizer
print("\n2. Testing Structure Visualizer...")
try:
    from src.paper_analysis.structure_visualizer import StructureVisualizer

    visualizer = StructureVisualizer()

    # Test SMILES to molecule
    aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    mol = visualizer.smiles_to_mol(aspirin_smiles)

    if mol:
        print(f"   ✓ Created molecule from SMILES")

        # Test property calculation
        props = visualizer.get_molecular_properties(aspirin_smiles)
        print(f"   ✓ Calculated properties: MW={props.get('molecular_weight', 0):.2f}")

        # Test drawing
        img_bytes = visualizer.draw_from_smiles(aspirin_smiles)
        if img_bytes:
            print(f"   ✓ Generated 2D structure ({len(img_bytes)} bytes)")
        else:
            print("   ⚠ Could not generate image")
    else:
        print("   ✗ Failed to create molecule")

except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 3: Summarizer
print("\n3. Testing Paper Summarizer...")
try:
    from src.paper_analysis.summarizer import PaperSummarizer
    import os

    summarizer = PaperSummarizer()

    if summarizer.client:
        print(f"   ✓ OpenAI client initialized (model: {summarizer.model})")

        sample_paper = """
        Abstract: This study investigates the potential of Metformin for cancer treatment.
        We conducted in vitro experiments on multiple cancer cell lines.

        Methods: Cancer cells were treated with Metformin at varying concentrations.
        Cell viability was assessed using MTT assays.

        Results: Metformin significantly reduced cancer cell proliferation at 10 mM.
        The IC50 was determined to be 5 mM.

        Conclusion: Metformin shows promise as a potential anticancer agent and warrants
        further investigation in clinical trials.
        """

        print("   ✓ Testing summary generation...")
        summary = summarizer.summarize_paper(sample_paper, max_length=200)

        if summary and summary.get('objective'):
            print(f"   ✓ Generated summary successfully")
            print(f"     Objective: {summary['objective'][:60]}...")
        else:
            print("   ⚠ Summary generated but structure unclear")

    else:
        print("   ⚠ OpenAI not configured (OPENAI_API_KEY not set)")
        print("     Fallback mode will be used")

except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 4: PDF Extractor (without actual PDF)
print("\n4. Testing PDF Extractor...")
try:
    from src.paper_analysis.pdf_extractor import PDFExtractor

    extractor = PDFExtractor()
    print(f"   ✓ PDF Extractor initialized")
    print(f"     Using: {'pdfplumber' if extractor.use_pdfplumber else 'PyPDF2'}")

except Exception as e:
    print(f"   ✗ Failed: {e}")
    print("     Install with: pip install pdfplumber PyPDF2")

# Summary
print("\n" + "=" * 70)
print("Testing Complete!")
print("=" * 70)
print("\nTo use the Paper Analysis feature:")
print("  1. Install dependencies: pip install pdfplumber PyPDF2 pubchempy")
print("  2. Set OPENAI_API_KEY in .env file")
print("  3. Run: streamlit run app/streamlit_app.py")
print("  4. Navigate to the 'Paper Analysis' tab")
print("\nSee PAPER_ANALYSIS_GUIDE.md for detailed instructions.")
