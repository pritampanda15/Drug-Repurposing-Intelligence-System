"""
Chemical structure visualization using RDKit.
"""

import logging
from typing import Optional, Tuple
from io import BytesIO
import base64

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem, Descriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

try:
    import pubchempy as pcp
    HAS_PUBCHEM = True
except ImportError:
    HAS_PUBCHEM = False

logger = logging.getLogger(__name__)


class StructureVisualizer:
    """Visualize chemical structures from names or SMILES."""

    def __init__(self):
        """Initialize structure visualizer."""
        if not HAS_RDKIT:
            raise ImportError("RDKit not installed. Install with: pip install rdkit")

        self.img_size = (400, 400)

    def name_to_smiles(self, chemical_name: str) -> Optional[str]:
        """
        Convert chemical name to SMILES.

        Parameters
        ----------
        chemical_name : str
            Chemical name

        Returns
        -------
        str or None
            SMILES string if found
        """
        if not HAS_PUBCHEM:
            logger.warning("PubChemPy not available. Cannot convert name to SMILES.")
            return None

        try:
            compounds = pcp.get_compounds(chemical_name, 'name', listkey_count=1)
            if compounds:
                # Use connectivity_smiles instead of deprecated canonical_smiles
                try:
                    return compounds[0].canonical_smiles  # Still works for now
                except AttributeError:
                    return compounds[0].isomeric_smiles
        except Exception as e:
            logger.debug(f"Failed to get SMILES for '{chemical_name}': {e}")

        return None

    def smiles_to_mol(self, smiles: str) -> Optional[object]:
        """
        Convert SMILES to RDKit molecule.

        Parameters
        ----------
        smiles : str
            SMILES string

        Returns
        -------
        Mol or None
            RDKit molecule object
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                AllChem.Compute2DCoords(mol)
                return mol
        except Exception as e:
            logger.error(f"Failed to create molecule from SMILES: {e}")

        return None

    def draw_molecule(
        self,
        mol: object,
        size: Tuple[int, int] = None
    ) -> Optional[bytes]:
        """
        Draw molecule as PNG image.

        Parameters
        ----------
        mol : Mol
            RDKit molecule object
        size : Tuple[int, int]
            Image size (width, height)

        Returns
        -------
        bytes or None
            PNG image data
        """
        if mol is None:
            return None

        size = size or self.img_size

        try:
            img = Draw.MolToImage(mol, size=size)

            # Convert to bytes
            buf = BytesIO()
            img.save(buf, format='PNG')
            return buf.getvalue()

        except Exception as e:
            logger.error(f"Failed to draw molecule: {e}")
            return None

    def draw_from_smiles(
        self,
        smiles: str,
        size: Tuple[int, int] = None
    ) -> Optional[bytes]:
        """
        Draw molecule from SMILES string.

        Parameters
        ----------
        smiles : str
            SMILES string
        size : Tuple[int, int]
            Image size

        Returns
        -------
        bytes or None
            PNG image data
        """
        mol = self.smiles_to_mol(smiles)
        return self.draw_molecule(mol, size)

    def draw_from_name(
        self,
        chemical_name: str,
        size: Tuple[int, int] = None
    ) -> Optional[bytes]:
        """
        Draw molecule from chemical name.

        Parameters
        ----------
        chemical_name : str
            Chemical name
        size : Tuple[int, int]
            Image size

        Returns
        -------
        bytes or None
            PNG image data
        """
        smiles = self.name_to_smiles(chemical_name)
        if not smiles:
            return None

        return self.draw_from_smiles(smiles, size)

    def get_molecular_properties(self, smiles: str) -> dict:
        """
        Calculate molecular properties from SMILES.

        Parameters
        ----------
        smiles : str
            SMILES string

        Returns
        -------
        dict
            Molecular properties
        """
        mol = self.smiles_to_mol(smiles)
        if not mol:
            return {}

        try:
            return {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'h_bond_donors': Descriptors.NumHDonors(mol),
                'h_bond_acceptors': Descriptors.NumHAcceptors(mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'aromatic_rings': Descriptors.NumAromaticRings(mol),
                'tpsa': Descriptors.TPSA(mol),
            }
        except Exception as e:
            logger.error(f"Failed to calculate properties: {e}")
            return {}

    def image_to_base64(self, img_bytes: bytes) -> str:
        """Convert image bytes to base64 string for display."""
        return base64.b64encode(img_bytes).decode()
