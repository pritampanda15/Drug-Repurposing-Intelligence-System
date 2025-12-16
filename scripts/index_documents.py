#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Index Documents for RAG

This script indexes biomedical documents for RAG-based explanations.

Usage:
    python scripts/index_documents.py [OPTIONS]
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.document_processor import DocumentProcessor


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def index_sample_data(processor: DocumentProcessor) -> None:
    """Index sample biomedical data."""
    logger.info("Indexing sample biomedical data...")

    # Create collection
    collection = processor.create_collection("drug_mechanisms")

    # Sample drug mechanism data
    sample_docs = [
        "Aspirin (acetylsalicylic acid) irreversibly inhibits cyclooxygenase-1 and cyclooxygenase-2 (COX-1 and COX-2) enzymes. "
        "This inhibition prevents the conversion of arachidonic acid to prostaglandins, which are mediators of inflammation, pain, and fever. "
        "Aspirin's antiplatelet effects result from inhibition of COX-1 in platelets, preventing thromboxane A2 synthesis.",

        "Metformin is a biguanide antidiabetic agent that primarily works by activating AMP-activated protein kinase (AMPK). "
        "This leads to reduced hepatic glucose production (gluconeogenesis), increased insulin sensitivity in peripheral tissues, "
        "and enhanced glucose uptake in skeletal muscle. Metformin also has beneficial effects on lipid metabolism.",

        "Imatinib is a tyrosine kinase inhibitor that specifically targets BCR-ABL fusion protein in chronic myeloid leukemia (CML). "
        "It also inhibits other kinases including c-KIT and PDGFR. Imatinib competitively binds to the ATP-binding site of these kinases, "
        "preventing phosphorylation of downstream substrates and blocking cellular proliferation.",

        "Statins (e.g., atorvastatin, simvastatin) competitively inhibit HMG-CoA reductase, the rate-limiting enzyme in cholesterol biosynthesis. "
        "This reduces hepatic cholesterol synthesis, leading to upregulation of LDL receptors and increased clearance of LDL cholesterol from plasma. "
        "Statins also have pleiotropic effects including anti-inflammatory and antioxidant properties.",

        "ACE inhibitors (e.g., lisinopril, enalapril) block angiotensin-converting enzyme, preventing conversion of angiotensin I to angiotensin II. "
        "This reduces vasoconstriction, aldosterone secretion, and sodium retention, leading to decreased blood pressure. "
        "ACE inhibitors are used in hypertension, heart failure, and diabetic nephropathy.",
    ]

    sample_metadata = [
        {
            "drug_id": "DB00945",
            "drug_name": "Aspirin",
            "class": "NSAID",
            "source": "drugbank",
            "indications": "pain, inflammation, cardiovascular protection"
        },
        {
            "drug_id": "DB00331",
            "drug_name": "Metformin",
            "class": "Biguanide",
            "source": "drugbank",
            "indications": "type 2 diabetes, PCOS"
        },
        {
            "drug_id": "DB00619",
            "drug_name": "Imatinib",
            "class": "Tyrosine kinase inhibitor",
            "source": "drugbank",
            "indications": "CML, GIST"
        },
        {
            "drug_id": "DB01076",
            "drug_name": "Atorvastatin",
            "class": "Statin",
            "source": "drugbank",
            "indications": "hypercholesterolemia, cardiovascular disease prevention"
        },
        {
            "drug_id": "DB00722",
            "drug_name": "Lisinopril",
            "class": "ACE inhibitor",
            "source": "drugbank",
            "indications": "hypertension, heart failure, post-MI"
        },
    ]

    # Add documents
    processor.add_documents(
        collection=collection,
        texts=sample_docs,
        metadatas=sample_metadata,
        ids=[f"drug_{m['drug_id']}" for m in sample_metadata]
    )

    logger.info(f"✓ Indexed {len(sample_docs)} drug mechanism documents")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Index documents for RAG system"
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='michiyasunaga/BioLinkBERT-base',
        help='Sentence transformer model for embeddings'
    )
    parser.add_argument(
        '--persist-dir',
        type=str,
        default='data/knowledge_base/chroma',
        help='Directory for ChromaDB persistence'
    )
    parser.add_argument(
        '--drugbank',
        type=str,
        default='data/raw/drugbank/full_database.xml',
        help='Path to DrugBank XML file'
    )
    parser.add_argument(
        '--pubmed',
        action='store_true',
        help='Enable PubMed literature indexing'
    )
    parser.add_argument(
        '--pubmed-queries',
        type=str,
        nargs='+',
        default=['drug repurposing', 'drug repositioning'],
        help='PubMed search queries'
    )
    parser.add_argument(
        '--skip-sample',
        action='store_true',
        help='Skip indexing sample data'
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Indexing Documents for RAG System")
    logger.info("=" * 60)

    try:
        # Initialize processor
        processor = DocumentProcessor(
            embedding_model=args.embedding_model,
            persist_directory=args.persist_dir
        )

        # Index sample data (unless skipped)
        if not args.skip_sample:
            index_sample_data(processor)

        # Index DrugBank if file exists
        if Path(args.drugbank).exists():
            logger.info(f"\n{'='*60}")
            logger.info("Processing DrugBank")
            logger.info(f"{'='*60}")
            processor.process_drugbank_data(
                drugbank_path=args.drugbank,
                collection_name="drug_mechanisms"
            )
        else:
            logger.warning(f"DrugBank XML not found at {args.drugbank}")
            logger.info("To use DrugBank:")
            logger.info("  1. Register at https://go.drugbank.com/releases/latest")
            logger.info("  2. Download full_database.xml")
            logger.info(f"  3. Save to {args.drugbank}")

        # Index PubMed if requested
        if args.pubmed:
            logger.info(f"\n{'='*60}")
            logger.info("Processing PubMed Literature")
            logger.info(f"{'='*60}")

            try:
                from src.rag.pubmed_fetcher import PubMedFetcher

                fetcher = PubMedFetcher()
                collection = processor.create_collection("pubmed_literature")

                total_articles = 0
                for query in args.pubmed_queries:
                    logger.info(f"Fetching articles for: {query}")
                    pmids = fetcher.search_drug_repurposing(query, max_results=50)

                    if pmids:
                        articles = fetcher.fetch_abstracts(pmids)
                        texts, metadatas, ids = fetcher.create_documents(articles)

                        if texts:
                            processor.add_documents(collection, texts, metadatas, ids)
                            total_articles += len(articles)
                            logger.info(f"  Added {len(articles)} articles")

                logger.info(f"✓ Indexed {total_articles} PubMed articles")

            except ImportError as e:
                logger.error(f"PubMed indexing failed: {e}")
                logger.info("Install BioPython: pip install biopython")
            except ValueError as e:
                logger.error(f"PubMed configuration error: {e}")
                logger.info("Set NCBI_EMAIL and optionally NCBI_API_KEY in .env")

        logger.info("\n" + "=" * 60)
        logger.info("✓ Document indexing complete!")
        logger.info(f"ChromaDB location: {args.persist_dir}")
        logger.info("\nIndexed Collections:")
        logger.info("  - drug_mechanisms: Drug information and mechanisms")
        if args.pubmed:
            logger.info("  - pubmed_literature: Scientific literature")
        logger.info("\nNext steps:")
        logger.info("  1. Use RAG for predictions: python -m src.rag.retriever")
        logger.info("  2. Launch web interface: streamlit run app/streamlit_app.py")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Indexing failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
