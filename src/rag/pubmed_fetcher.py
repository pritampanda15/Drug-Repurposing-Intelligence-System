"""
PubMed Literature Fetcher

Fetches drug repurposing literature from PubMed using NCBI Entrez API.
"""

from typing import List, Dict, Optional
import time
import logging
import os

try:
    from Bio import Entrez
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    logging.warning("BioPython not installed. PubMed fetching disabled.")

logger = logging.getLogger(__name__)


class PubMedFetcher:
    """
    Fetch drug repurposing literature from PubMed.
    
    Uses NCBI Entrez API to search and retrieve abstracts.
    """
    
    def __init__(
        self,
        email: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize PubMed fetcher.
        
        Parameters
        ----------
        email : Optional[str]
            Email for NCBI (required by NCBI)
        api_key : Optional[str]
            NCBI API key (optional, increases rate limit)
        """
        if not BIOPYTHON_AVAILABLE:
            raise ImportError(
                "BioPython required for PubMed fetching. "
                "Install with: pip install biopython"
            )
        
        # Get credentials from environment if not provided
        self.email = email or os.getenv('NCBI_EMAIL')
        self.api_key = api_key or os.getenv('NCBI_API_KEY')
        
        if not self.email:
            raise ValueError(
                "Email required for NCBI Entrez. "
                "Set NCBI_EMAIL environment variable or pass email parameter."
            )
        
        # Configure Entrez
        Entrez.email = self.email
        if self.api_key:
            Entrez.api_key = self.api_key
            logger.info("Using NCBI API key (10 requests/second)")
        else:
            logger.info("No API key - rate limited to 3 requests/second")
        
        # Rate limiting
        self.requests_per_second = 10 if self.api_key else 3
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.requests_per_second
        
        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def search_drug_disease(
        self,
        drug_name: str,
        disease_name: str,
        max_results: int = 20
    ) -> List[str]:
        """
        Search PubMed for drug-disease literature.
        
        Parameters
        ----------
        drug_name : str
            Drug name
        disease_name : str
            Disease name
        max_results : int
            Maximum number of results
        
        Returns
        -------
        List[str]
            List of PubMed IDs (PMIDs)
        """
        # Construct query
        query = f'("{drug_name}"[Title/Abstract]) AND ("{disease_name}"[Title/Abstract]) AND (drug repurposing OR drug repositioning OR therapeutic use)'
        
        logger.info(f"Searching PubMed: {query}")
        
        try:
            self._rate_limit()
            
            handle = Entrez.esearch(
                db='pubmed',
                term=query,
                retmax=max_results,
                sort='relevance'
            )
            
            results = Entrez.read(handle)
            handle.close()
            
            pmids = results.get('IdList', [])
            logger.info(f"Found {len(pmids)} articles")
            
            return pmids
        
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return []
    
    def search_drug_repurposing(
        self,
        drug_name: str,
        max_results: int = 10
    ) -> List[str]:
        """
        Search for general drug repurposing literature.
        
        Parameters
        ----------
        drug_name : str
            Drug name
        max_results : int
            Maximum number of results
        
        Returns
        -------
        List[str]
            List of PubMed IDs
        """
        query = f'("{drug_name}"[Title/Abstract]) AND (drug repurposing OR drug repositioning OR repurposed)'
        
        logger.info(f"Searching PubMed: {query}")
        
        try:
            self._rate_limit()
            
            handle = Entrez.esearch(
                db='pubmed',
                term=query,
                retmax=max_results,
                sort='relevance'
            )
            
            results = Entrez.read(handle)
            handle.close()
            
            return results.get('IdList', [])
        
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return []
    
    def fetch_abstracts(
        self,
        pmids: List[str],
        batch_size: int = 100
    ) -> List[Dict]:
        """
        Fetch article abstracts from PubMed.
        
        Parameters
        ----------
        pmids : List[str]
            List of PubMed IDs
        batch_size : int
            Number of articles to fetch per request
        
        Returns
        -------
        List[Dict]
            List of article dictionaries with title, abstract, etc.
        """
        articles = []
        
        # Process in batches
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i + batch_size]
            
            try:
                self._rate_limit()
                
                handle = Entrez.efetch(
                    db='pubmed',
                    id=','.join(batch_pmids),
                    rettype='medline',
                    retmode='xml'
                )
                
                records = Entrez.read(handle)
                handle.close()
                
                # Extract article info
                for article in records.get('PubmedArticle', []):
                    article_data = self._extract_article_data(article)
                    if article_data:
                        articles.append(article_data)
                
            except Exception as e:
                logger.error(f"Failed to fetch batch: {e}")
                continue
        
        logger.info(f"Fetched {len(articles)} abstracts")
        return articles
    
    def _extract_article_data(self, article) -> Optional[Dict]:
        """Extract relevant data from PubMed article."""
        try:
            medline = article.get('MedlineCitation', {})
            pubmed_data = article.get('PubmedData', {})
            
            # PMID
            pmid = str(medline.get('PMID', ''))
            
            # Article details
            article_info = medline.get('Article', {})
            
            # Title
            title = article_info.get('ArticleTitle', '')
            
            # Abstract
            abstract_parts = article_info.get('Abstract', {}).get('AbstractText', [])
            if isinstance(abstract_parts, list):
                abstract = ' '.join([str(part) for part in abstract_parts])
            else:
                abstract = str(abstract_parts)
            
            if not abstract:
                return None
            
            # Journal
            journal = article_info.get('Journal', {}).get('Title', '')
            
            # Publication date
            pub_date = article_info.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
            year = pub_date.get('Year', '')
            
            # Authors
            authors = []
            author_list = article_info.get('AuthorList', [])
            for author in author_list[:3]:  # First 3 authors
                last_name = author.get('LastName', '')
                if last_name:
                    authors.append(last_name)
            
            return {
                'pmid': pmid,
                'title': title,
                'abstract': abstract,
                'journal': journal,
                'year': year,
                'authors': ', '.join(authors)
            }
        
        except Exception as e:
            logger.error(f"Error extracting article data: {e}")
            return None
    
    def create_documents(
        self,
        articles: List[Dict],
        drug_name: Optional[str] = None,
        disease_name: Optional[str] = None
    ) -> tuple:
        """
        Create documents for RAG indexing.
        
        Parameters
        ----------
        articles : List[Dict]
            List of article dictionaries
        drug_name : Optional[str]
            Drug name for metadata
        disease_name : Optional[str]
            Disease name for metadata
        
        Returns
        -------
        tuple
            (texts, metadatas, ids)
        """
        texts = []
        metadatas = []
        ids = []
        
        for article in articles:
            # Create text
            text = f"Title: {article['title']}\n\n"
            text += f"Abstract: {article['abstract']}\n\n"
            text += f"Authors: {article['authors']}\n"
            text += f"Journal: {article['journal']} ({article['year']})\n"
            text += f"PMID: {article['pmid']}"
            
            texts.append(text)
            
            # Metadata
            metadata = {
                'pmid': article['pmid'],
                'title': article['title'],
                'journal': article['journal'],
                'year': article['year'],
                'source': 'pubmed'
            }
            
            if drug_name:
                metadata['drug_name'] = drug_name
            if disease_name:
                metadata['disease_name'] = disease_name
            
            metadatas.append(metadata)
            ids.append(f"pubmed_{article['pmid']}")
        
        return texts, metadatas, ids
