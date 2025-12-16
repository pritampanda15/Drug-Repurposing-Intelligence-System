"""
DrugBank XML Parser

Parses DrugBank full database XML to extract drug information
for RAG system.
"""

from pathlib import Path
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DrugBankParser:
    """
    Parse DrugBank XML database.
    
    Extracts drug mechanisms, indications, and other relevant information
    for the RAG knowledge base.
    """
    
    # DrugBank XML namespace
    NS = {'db': 'http://www.drugbank.ca'}
    
    def __init__(self, xml_path: str):
        """
        Initialize DrugBank parser.
        
        Parameters
        ----------
        xml_path : str
            Path to DrugBank full_database.xml file
        """
        self.xml_path = Path(xml_path)
        
        if not self.xml_path.exists():
            raise FileNotFoundError(
                f"DrugBank XML not found at {xml_path}. "
                "Download from https://go.drugbank.com/releases/latest"
            )
        
        logger.info(f"Initialized DrugBank parser for {xml_path}")
    
    def parse(self) -> List[Dict]:
        """
        Parse DrugBank XML and extract drug information.
        
        Returns
        -------
        List[Dict]
            List of drug dictionaries with extracted information
        """
        logger.info("Parsing DrugBank XML (this may take a few minutes)...")
        
        drugs = []
        
        # Parse XML iteratively to save memory
        context = ET.iterparse(self.xml_path, events=('start', 'end'))
        context = iter(context)
        event, root = next(context)
        
        drug_count = 0
        
        for event, elem in tqdm(context, desc="Parsing DrugBank"):
            if event == 'end' and elem.tag == f"{{{self.NS['db']}}}drug":
                drug_data = self._extract_drug_data(elem)
                
                if drug_data:
                    drugs.append(drug_data)
                    drug_count += 1
                
                # Clear element to save memory
                elem.clear()
                root.clear()
        
        logger.info(f"Parsed {drug_count} drugs from DrugBank")
        return drugs
    
    def _extract_drug_data(self, drug_elem) -> Optional[Dict]:
        """
        Extract relevant data from a drug XML element.
        
        Parameters
        ----------
        drug_elem : xml.etree.ElementTree.Element
            Drug XML element
        
        Returns
        -------
        Optional[Dict]
            Drug data dictionary or None if not a small molecule
        """
        # Only process small molecule drugs
        drug_type = drug_elem.get('type')
        if drug_type != 'small molecule':
            return None
        
        # DrugBank ID
        drugbank_id = self._get_text(
            drug_elem, 
            './/db:drugbank-id[@primary="true"]'
        )
        
        if not drugbank_id:
            return None
        
        # Basic information
        name = self._get_text(drug_elem, './db:name')
        description = self._get_text(drug_elem, './db:description')
        
        # Mechanism of action
        mechanism = self._get_text(
            drug_elem, 
            './db:pharmacodynamics'
        ) or self._get_text(
            drug_elem,
            './db:mechanism-of-action'
        )
        
        # Indications
        indication = self._get_text(drug_elem, './db:indication')
        
        # Pharmacology
        pharmacodynamics = self._get_text(
            drug_elem, 
            './db:pharmacodynamics'
        )
        
        # Drug class
        categories = self._get_categories(drug_elem)
        
        # Groups (approved, experimental, etc.)
        groups = self._get_groups(drug_elem)
        
        # Only include if has useful text
        if not (mechanism or indication or pharmacodynamics):
            return None
        
        return {
            'drugbank_id': drugbank_id,
            'name': name,
            'description': description,
            'mechanism': mechanism,
            'indication': indication,
            'pharmacodynamics': pharmacodynamics,
            'categories': categories,
            'groups': groups
        }
    
    def _get_text(self, elem, xpath: str) -> Optional[str]:
        """Get text content from XML element."""
        found = elem.find(xpath, self.NS)
        if found is not None and found.text:
            return found.text.strip()
        return None
    
    def _get_categories(self, drug_elem) -> List[str]:
        """Extract drug categories."""
        categories = []
        cat_elems = drug_elem.findall('.//db:category', self.NS)
        
        for cat_elem in cat_elems:
            cat_name = self._get_text(cat_elem, './db:category')
            if cat_name:
                categories.append(cat_name)
        
        return categories[:5]  # Limit to top 5
    
    def _get_groups(self, drug_elem) -> List[str]:
        """Extract drug groups (approved, experimental, etc.)."""
        groups = []
        group_elems = drug_elem.findall('.//db:group', self.NS)
        
        for group_elem in group_elems:
            if group_elem.text:
                groups.append(group_elem.text.strip())
        
        return groups
    
    def create_documents(self, drugs: List[Dict]) -> tuple:
        """
        Create documents and metadata for RAG indexing.
        
        Parameters
        ----------
        drugs : List[Dict]
            List of drug dictionaries from parse()
        
        Returns
        -------
        tuple
            (texts, metadatas, ids) for document indexing
        """
        texts = []
        metadatas = []
        ids = []
        
        for drug in drugs:
            # Create mechanism document
            if drug.get('mechanism'):
                text = self._create_mechanism_text(drug)
                texts.append(text)
                metadatas.append({
                    'drug_id': drug['drugbank_id'],
                    'drug_name': drug['name'],
                    'source': 'drugbank',
                    'type': 'mechanism',
                    'categories': ','.join(drug.get('categories', [])),
                    'groups': ','.join(drug.get('groups', []))
                })
                ids.append(f"drugbank_{drug['drugbank_id']}_mechanism")
            
            # Create indication document
            if drug.get('indication'):
                text = self._create_indication_text(drug)
                texts.append(text)
                metadatas.append({
                    'drug_id': drug['drugbank_id'],
                    'drug_name': drug['name'],
                    'source': 'drugbank',
                    'type': 'indication',
                    'groups': ','.join(drug.get('groups', []))
                })
                ids.append(f"drugbank_{drug['drugbank_id']}_indication")
        
        logger.info(f"Created {len(texts)} documents from {len(drugs)} drugs")
        return texts, metadatas, ids
    
    def _create_mechanism_text(self, drug: Dict) -> str:
        """Create mechanism of action text."""
        parts = [f"Drug: {drug['name']} ({drug['drugbank_id']})"]
        
        if drug.get('description'):
            parts.append(f"Description: {drug['description']}")
        
        if drug.get('mechanism'):
            parts.append(f"Mechanism of Action: {drug['mechanism']}")
        
        if drug.get('pharmacodynamics'):
            parts.append(f"Pharmacodynamics: {drug['pharmacodynamics']}")
        
        if drug.get('categories'):
            parts.append(f"Categories: {', '.join(drug['categories'])}")
        
        return "\n\n".join(parts)
    
    def _create_indication_text(self, drug: Dict) -> str:
        """Create indication text."""
        parts = [f"Drug: {drug['name']} ({drug['drugbank_id']})"]
        
        if drug.get('indication'):
            parts.append(f"Indications: {drug['indication']}")
        
        return "\n\n".join(parts)
