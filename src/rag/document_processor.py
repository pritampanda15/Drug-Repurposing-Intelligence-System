"""
Document processing for RAG system.

Handles document chunking, embedding, and indexing.
"""

from pathlib import Path
from typing import List, Dict, Optional
import logging

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Process and index documents for RAG retrieval.
    """

    def __init__(
        self,
        embedding_model: str = "michiyasunaga/BioLinkBERT-base",
        persist_directory: str = "data/knowledge_base/chroma"
    ):
        """
        Initialize document processor.

        Parameters
        ----------
        embedding_model : str
            Name of sentence transformer model
        persist_directory : str
            Directory for ChromaDB persistence
        """
        self.embedding_model_name = embedding_model

        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize ChromaDB
        persist_path = Path(persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(persist_path))

        logger.info(f"Initialized ChromaDB at {persist_directory}")

    def create_collection(self, name: str) -> chromadb.Collection:
        """
        Create or get a collection.

        Parameters
        ----------
        name : str
            Collection name

        Returns
        -------
        chromadb.Collection
            Collection object
        """
        try:
            collection = self.client.get_or_create_collection(name=name)
            logger.info(f"Collection '{name}' ready (size: {collection.count()})")
            return collection
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 512,
        overlap: int = 50
    ) -> List[str]:
        """
        Split text into chunks.

        Parameters
        ----------
        text : str
            Text to chunk
        chunk_size : int
            Maximum chunk size in characters
        overlap : int
            Overlap between chunks

        Returns
        -------
        List[str]
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                if last_period > chunk_size * 0.5:
                    end = start + last_period + 1
                    chunk = text[start:end]

            chunks.append(chunk.strip())
            start = end - overlap

        return chunks

    def add_documents(
        self,
        collection: chromadb.Collection,
        texts: List[str],
        metadatas: List[Dict],
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add documents to collection.

        Parameters
        ----------
        collection : chromadb.Collection
            Target collection
        texts : List[str]
            Document texts
        metadatas : List[Dict]
            Metadata for each document
        ids : Optional[List[str]]
            Document IDs (auto-generated if None)
        """
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} documents...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

        # Add to collection
        collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

        logger.info(f"Added {len(texts)} documents to collection")

    def process_drugbank_data(
        self,
        drugbank_path: str,
        collection_name: str = "drug_mechanisms"
    ) -> None:
        """
        Process DrugBank XML database.

        Parameters
        ----------
        drugbank_path : str
            Path to DrugBank full_database.xml file
        collection_name : str
            Name of collection to create
        """
        from ..data.drugbank_parser import DrugBankParser

        collection = self.create_collection(collection_name)

        logger.info(f"Processing DrugBank from {drugbank_path}")

        # Parse DrugBank XML
        parser = DrugBankParser(drugbank_path)
        drugs = parser.parse()

        # Create documents
        texts, metadatas, ids = parser.create_documents(drugs)

        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]

            self.add_documents(
                collection,
                batch_texts,
                batch_metadatas,
                batch_ids
            )

            logger.info(f"Indexed {min(i + batch_size, len(texts))}/{len(texts)} drugs")

        logger.info(f"✓ DrugBank indexing complete: {len(drugs)} drugs, {len(texts)} documents")
