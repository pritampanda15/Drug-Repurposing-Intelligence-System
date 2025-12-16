#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download Data Script for Drug Repurposing Intelligence System

This script downloads the required datasets:
1. Hetionet knowledge graph (automatic)
2. DrugBank (requires manual download with academic license)

Usage:
    python scripts/download_data.py [--data-dir DATA_DIR]
"""

import argparse
import bz2
import logging
from pathlib import Path
from typing import Optional
import sys

import requests
from tqdm import tqdm
import yaml


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataDownloader:
    """Download and setup datasets for drug repurposing system."""

    HETIONET_URL = "https://github.com/hetio/hetionet/raw/main/hetnet/json/hetionet-v1.0.json.bz2"

    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize data downloader.

        Parameters
        ----------
        data_dir : str
            Root directory for storing raw data
        """
        self.data_dir = Path(data_dir)
        self.hetionet_dir = self.data_dir / "hetionet"
        self.drugbank_dir = self.data_dir / "drugbank"
        self.pubmed_dir = self.data_dir / "pubmed_abstracts"

        # Create directories
        self.hetionet_dir.mkdir(parents=True, exist_ok=True)
        self.drugbank_dir.mkdir(parents=True, exist_ok=True)
        self.pubmed_dir.mkdir(parents=True, exist_ok=True)

    def download_file(
        self,
        url: str,
        output_path: Path,
        decompress_bz2: bool = True
    ) -> bool:
        """
        Download a file from URL with progress bar.

        Parameters
        ----------
        url : str
            URL to download from
        output_path : Path
            Path to save the file
        decompress_bz2 : bool
            If True and file is .bz2, decompress it

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            logger.info(f"Downloading from {url}")

            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            # Temporary file for compressed data
            temp_path = output_path.with_suffix(output_path.suffix + '.bz2') if decompress_bz2 and not url.endswith('.bz2') else output_path
            if url.endswith('.bz2') and decompress_bz2:
                temp_path = output_path.with_suffix(output_path.suffix + '.bz2')

            # Download
            with open(temp_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            # Decompress if needed
            if decompress_bz2 and str(temp_path).endswith('.bz2'):
                logger.info(f"Decompressing {temp_path.name}")

                with bz2.open(temp_path, 'rb') as f_in:
                    with open(output_path, 'wb') as f_out:
                        # Read and write in chunks
                        while True:
                            chunk = f_in.read(8192)
                            if not chunk:
                                break
                            f_out.write(chunk)

                # Remove compressed file
                temp_path.unlink()
                logger.info(f"Decompressed to {output_path}")

            logger.info(f"Successfully downloaded to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return False

    def download_hetionet(self) -> bool:
        """
        Download Hetionet knowledge graph.

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        output_path = self.hetionet_dir / "hetionet-v1.0.json"

        if output_path.exists():
            logger.info(f"Hetionet already exists at {output_path}")
            return True

        logger.info("Downloading Hetionet knowledge graph...")
        logger.info("Source: https://het.io")
        logger.info(f"Nodes: 47,031 | Edges: 2,250,197")

        success = self.download_file(
            self.HETIONET_URL,
            output_path,
            decompress_bz2=True
        )

        if success:
            logger.info("✓ Hetionet download complete")

        return success

    def check_drugbank(self) -> None:
        """
        Check for DrugBank data and provide instructions if missing.
        """
        drugbank_xml = self.drugbank_dir / "full_database.xml"

        if drugbank_xml.exists():
            logger.info(f"✓ DrugBank found at {drugbank_xml}")
        else:
            logger.warning("✗ DrugBank not found")
            logger.info("\nDrugBank requires an academic license:")
            logger.info("1. Register at https://go.drugbank.com/releases/latest")
            logger.info("2. Download the 'Full Database' XML file")
            logger.info(f"3. Save it to: {drugbank_xml}")
            logger.info("\nNote: DrugBank is optional for initial model training")
            logger.info("      It's primarily used for RAG explanations")

    def create_readme_files(self) -> None:
        """Create README files in data directories with information."""

        # Hetionet README
        hetionet_readme = self.hetionet_dir / "README.md"
        hetionet_readme.write_text("""# Hetionet Data

This directory contains the Hetionet knowledge graph.

## About Hetionet

- **Version**: v1.0
- **Source**: https://het.io
- **License**: CC0 1.0 (Public Domain)
- **Citation**: Himmelstein et al. (2017), eLife

## Contents

- `hetionet-v1.0.json`: Complete heterogeneous biomedical knowledge graph
  - 47,031 nodes across 11 types
  - 2,250,197 edges across 24 types

## Node Types

Compound, Disease, Gene, Anatomy, Pathway, Side Effect, Pharmacologic Class,
Biological Process, Cellular Component, Molecular Function, Symptom

## Key Edge Types

- Compound-treats-Disease
- Compound-palliates-Disease
- Compound-targets-Gene
- Disease-associates-Gene
- Gene-participates-Pathway
""")

        # DrugBank README
        drugbank_readme = self.drugbank_dir / "README.md"
        drugbank_readme.write_text("""# DrugBank Data

This directory should contain DrugBank data files.

## How to Obtain

DrugBank requires an academic license:

1. Visit https://go.drugbank.com/releases/latest
2. Create an account with your academic email
3. Download the "Full Database" XML file
4. Save as `full_database.xml` in this directory

## License

DrugBank is free for academic research but requires registration.
Commercial use requires a separate license.

## What We Use

- Drug mechanisms of action
- Pharmacodynamics descriptions
- Known indications
- Drug-target relationships

This data is used primarily for RAG-based explanations, not for model training.
""")

        # PubMed README
        pubmed_readme = self.pubmed_dir / "README.md"
        pubmed_readme.write_text("""# PubMed Abstracts

This directory stores PubMed abstracts retrieved for RAG.

## Retrieval

Abstracts are downloaded automatically during RAG indexing using the NCBI E-utilities API.

## Requirements

Set these environment variables in `.env`:
- `NCBI_API_KEY`: Your NCBI API key (optional but recommended)
- `NCBI_EMAIL`: Your email address (required)

## API Key

Get a key at: https://www.ncbi.nlm.nih.gov/account/settings/

With an API key, you can make 10 requests/second instead of 3 requests/second.

## Usage

PubMed abstracts are queried on-demand for specific drug-disease pairs
to provide literature evidence for predictions.
""")

        logger.info("Created README files in data directories")

    def run(self) -> bool:
        """
        Run the complete download process.

        Returns
        -------
        bool
            True if all critical downloads succeeded
        """
        logger.info("=" * 60)
        logger.info("Drug Repurposing Intelligence System - Data Download")
        logger.info("=" * 60)

        success = True

        # Download Hetionet (required)
        if not self.download_hetionet():
            success = False

        # Check DrugBank (optional)
        self.check_drugbank()

        # Create documentation
        self.create_readme_files()

        logger.info("\n" + "=" * 60)
        if success:
            logger.info("✓ Data download complete!")
            logger.info("\nNext steps:")
            logger.info("  1. python scripts/build_knowledge_graph.py")
            logger.info("  2. python scripts/train_model.py")
        else:
            logger.info("✗ Some downloads failed. Check logs above.")

        logger.info("=" * 60)

        return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download datasets for drug repurposing system"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw',
        help='Directory for storing raw data (default: data/raw)'
    )

    args = parser.parse_args()

    downloader = DataDownloader(data_dir=args.data_dir)
    success = downloader.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
