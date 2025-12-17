"""
RAG retrieval system for drug repurposing explanations.
"""

from typing import List, Dict, Optional
import logging

from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path

logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    Retrieve relevant documents for drug repurposing predictions.
    """

    def __init__(
        self,
        embedding_model: str = "michiyasunaga/BioLinkBERT-base",
        persist_directory: str = "data/knowledge_base/chroma"
    ):
        """
        Initialize retriever.

        Parameters
        ----------
        embedding_model : str
            Name of sentence transformer model
        persist_directory : str
            Directory for ChromaDB (relative to project root or absolute path)
        """
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        # Handle both relative and absolute paths
        persist_path = Path(persist_directory)
        if not persist_path.is_absolute():
            # Relative path - resolve from project root
            base_dir = Path(__file__).resolve().parents[2]  # src/rag/ -> project root
            persist_path = (base_dir / persist_directory).resolve()

        logger.info(f"Using ChromaDB directory: {persist_path}")

        self.client = chromadb.PersistentClient(path=str(persist_path))

    def retrieve(
        self,
        query: str,
        collection_name: str = "drug_mechanisms",
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Retrieve relevant documents.

        Parameters
        ----------
        query : str
            Query text
        collection_name : str
            Collection to search
        top_k : int
            Number of results to return
        filters : Optional[Dict]
            Metadata filters

        Returns
        -------
        List[Dict]
            Retrieved documents with metadata
        """
        try:
            collection = self.client.get_collection(collection_name)
        except Exception as e:
            logger.error(f"Collection '{collection_name}' not found: {e}")
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]

        # Search
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filters
        )

        # Format results
        documents = []
        for i in range(len(results['documents'][0])):
            documents.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })

        return documents

    def retrieve_for_prediction(
        self,
        drug_name: str,
        disease_name: str,
        top_k: int = 5
    ) -> Dict[str, List[Dict]]:
        """
        Retrieve context for a drug-disease prediction.

        Parameters
        ----------
        drug_name : str
            Drug name
        disease_name : str
            Disease name
        top_k : int
            Number of results per query

        Returns
        -------
        Dict[str, List[Dict]]
            Retrieved documents organized by type
        """
        results = {}

        # Query for drug mechanism
        drug_query = f"mechanism of action for {drug_name}"
        results['drug_mechanism'] = self.retrieve(
            drug_query,
            collection_name="drug_mechanisms",
            top_k=top_k
        )

        # Query for disease information
        disease_query = f"pathophysiology of {disease_name}"
        results['disease_info'] = self.retrieve(
            disease_query,
            collection_name="drug_mechanisms",
            top_k=top_k
        )

        # Query for related literature
        combined_query = f"{drug_name} treatment for {disease_name}"
        results['literature'] = self.retrieve(
            combined_query,
            collection_name="drug_mechanisms",
            top_k=top_k
        )

        return results
