"""
Chemical entity extraction from scientific text.
"""

import re
import logging
from typing import List, Dict, Set
from collections import Counter

try:
    import pubchempy as pcp
    HAS_PUBCHEM = True
except ImportError:
    HAS_PUBCHEM = False

logger = logging.getLogger(__name__)


class ChemicalExtractor:
    """Extract chemical entities from scientific text."""

    def __init__(self):
        """Initialize chemical extractor."""
        # Improved chemical name patterns with higher specificity
        self.patterns = [
            # Common drug suffixes (most specific first)
            r'\b\w{3,}(?:azole|oxacin|floxacin|mycin|cillin|cycline|phylline)\b',  # Antibiotics
            r'\b\w{3,}(?:olol|dipine|pril|sartan|statin|fibrate)\b',  # Cardiovascular
            r'\b\w{3,}(?:azepam|azolam|barbital)\b',  # Sedatives
            r'\b\w{3,}(?:triptan|caine|tani?l|morphine|codeine)\b',  # Pain/migraine (includes fentanyl/fentanil)
            r'\b\w{3,}(?:prazole|tidine|picam)\b',  # GI drugs
            r'\b\w{3,}(?:mab|ximab|zumab|umab)\b',  # Monoclonal antibodies
            r'\b\w{3,}(?:profen|phen|pirin)\b',  # NSAIDs (ibuprofen, acetaminophen, aspirin)
            r'\b\w{3,}(?:ine|ide|ate|ole|osin)\b',  # General suffixes
            # Common drug name patterns
            r'\b[A-Z][a-z]{3,}(?:in|ol|an|one|ide|ate|il|al)\b',  # Capitalized drugs
        ]

        # Expanded stopwords - common non-chemical words
        self.stopwords = {
            # Common words
            'the', 'and', 'for', 'with', 'from', 'that', 'this', 'these', 'those',
            'were', 'have', 'been', 'their', 'than', 'into', 'between', 'which',
            'using', 'during', 'after', 'before', 'within', 'about', 'through',
            # Paper sections
            'Figure', 'Table', 'Supplementary', 'Materials', 'Methods', 'Results',
            'Discussion', 'Introduction', 'Conclusion', 'References', 'Abstract',
            # Journal/publication terms
            'Journal', 'Science', 'Nature', 'Cell', 'Proceedings', 'Medicine',
            'Clinical', 'Research', 'Studies', 'Study', 'Trial', 'Trials',
            # Generic terms often matching patterns
            'patients', 'treatment', 'control', 'baseline', 'outcome', 'concentration',
            'administration', 'associated', 'comparison', 'significantly', 'effective',
            'analysis', 'protein', 'receptor', 'expression', 'inhibition', 'activation',
            'Protein', 'Receptor', 'Expression', 'Inhibition', 'Activation',
            'Design', 'Objective', 'Background', 'Setting', 'Intervention',
            # Common false positives
            'Section', 'Data', 'Group', 'Total', 'Mean', 'Median', 'Rate',
        }

    def extract_chemicals(
        self,
        text: str,
        min_length: int = 4,
        max_results: int = 50,
        validate: bool = False
    ) -> List[Dict[str, any]]:
        """
        Extract chemical entities from text.

        Parameters
        ----------
        text : str
            Input text
        min_length : int
            Minimum length for chemical names
        max_results : int
            Maximum number of results to return
        validate : bool
            Whether to validate with PubChem (slower)

        Returns
        -------
        List[Dict]
            List of extracted chemicals with metadata
        """
        # Extract candidate chemicals
        candidates = set()

        for pattern in self.patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            candidates.update(matches)

        # Filter candidates
        filtered = self._filter_candidates(candidates, min_length)

        if not filtered:
            logger.info("No chemicals found after filtering")
            return []

        # Count frequencies - use case-insensitive matching
        chemical_counts = Counter()
        for chem in filtered:
            # Count occurrences (case-insensitive)
            pattern = re.compile(r'\b' + re.escape(chem) + r'\b', re.IGNORECASE)
            count = len(pattern.findall(text))
            if count > 0:
                chemical_counts[chem] = count

        # Build results
        results = []
        for chemical, count in chemical_counts.most_common(max_results):
            entry = {
                'name': chemical,
                'count': count,
                'validated': False,
                'smiles': None,
                'inchi': None,
                'cid': None
            }

            # Validate with PubChem if requested
            if validate and HAS_PUBCHEM:
                pub_data = self._validate_with_pubchem(chemical)
                if pub_data:
                    entry.update(pub_data)
                    entry['validated'] = True

            results.append(entry)

        return results

    def _filter_candidates(self, candidates: Set[str], min_length: int) -> Set[str]:
        """Filter out non-chemical candidates with balanced heuristics."""
        filtered = set()

        for candidate in candidates:
            # Skip if too short
            if len(candidate) < min_length:
                continue

            # Skip stopwords (case-insensitive)
            if candidate.lower() in self.stopwords:
                continue

            # Skip if all uppercase (likely acronym)
            if candidate.isupper():
                continue

            # Skip very common general terms
            candidate_lower = candidate.lower()

            # Skip words ending in -ation/-tion (usually not drugs)
            if candidate_lower.endswith(('ation', 'tion')):
                continue

            # Skip if it's a common verb/adjective ending
            if candidate_lower.endswith(('ing', 'ed', 'ly')):
                continue

            # If it passed the pattern matching, it's likely a chemical
            # Just filter out obvious non-chemicals
            filtered.add(candidate)

        return filtered

    def _validate_with_pubchem(self, chemical_name: str) -> Dict[str, any]:
        """
        Validate chemical with PubChem.

        Parameters
        ----------
        chemical_name : str
            Chemical name to validate

        Returns
        -------
        Dict or None
            PubChem data if found
        """
        try:
            compounds = pcp.get_compounds(chemical_name, 'name')

            if compounds:
                compound = compounds[0]
                return {
                    'smiles': compound.canonical_smiles,
                    'inchi': compound.inchi,
                    'cid': compound.cid,
                    'iupac_name': compound.iupac_name,
                    'molecular_formula': compound.molecular_formula,
                    'molecular_weight': compound.molecular_weight
                }

        except Exception as e:
            logger.debug(f"PubChem validation failed for '{chemical_name}': {e}")

        return None

    def extract_drug_names(self, text: str, drug_database: List[str] = None) -> List[str]:
        """
        Extract known drug names from text.

        Parameters
        ----------
        text : str
            Input text
        drug_database : List[str]
            List of known drug names to search for

        Returns
        -------
        List[str]
            Found drug names
        """
        if not drug_database:
            # Use basic pattern matching
            return self.extract_chemicals(text, validate=False)

        # Search for known drugs
        found_drugs = []
        text_lower = text.lower()

        for drug in drug_database:
            if re.search(r'\b' + re.escape(drug.lower()) + r'\b', text_lower):
                found_drugs.append(drug)

        return found_drugs
