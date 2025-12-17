#!/bin/bash
# Quick installation script for Paper Analysis feature

echo "Installing Paper Analysis dependencies..."

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "ERROR: Virtual environment not activated!"
    echo "Please run: source .venv/bin/activate"
    exit 1
fi

echo "Installing packages..."
pip install pdfplumber PyPDF2 pubchempy httpx certifi

echo ""
echo "✅ Installation complete!"
echo ""
echo "To use the Paper Analysis feature:"
echo "  1. Make sure OPENAI_API_KEY is set in .env"
echo "  2. Run: streamlit run app/streamlit_app.py"
echo "  3. Navigate to the '📄 Paper Analysis' tab"
echo ""
echo "See PAPER_ANALYSIS_GUIDE.md for detailed instructions."
