"""
Drug Repurposing Predictor

High-level interface for making drug repurposing predictions.
"""

from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging

import torch
import yaml

from ..data.hetionet_loader import HetionetLoader
from ..models.link_predictor import DrugRepurposingModel
from ..rag.retriever import RAGRetriever
from ..rag.explanation_generator import ExplanationGenerator


logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """
    Result of a drug repurposing prediction.
    
    Attributes
    ----------
    drug_id : str
        Drug identifier
    drug_name : str
        Drug name
    disease_id : str
        Disease identifier
    disease_name : str
        Disease name
    prediction_score : float
        Prediction score (0-1)
    confidence : str
        Confidence level (low/medium/high)
    explanation : Optional[str]
        Natural language explanation
    """
    drug_id: str
    drug_name: str
    disease_id: str
    disease_name: str
    prediction_score: float
    confidence: str
    explanation: Optional[str] = None


class DrugRepurposingPredictor:
    """
    High-level interface for drug repurposing predictions.
    
    This class provides a simple API for making predictions and
    generating explanations for drug-disease pairs.
    """
    
    def __init__(
        self,
        model_path: str = "models/checkpoints/best_model.pt",
        config_path: str = "config/model_config.yaml",
        data_dir: str = "data/raw/hetionet",
        device: str = "cpu",
        enable_rag: bool = True
    ):
        """
        Initialize predictor.
        
        Parameters
        ----------
        model_path : str
            Path to trained model checkpoint
        config_path : str
            Path to model configuration
        data_dir : str
            Path to Hetionet data directory
        device : str
            Device to run model on ('cpu' or 'cuda')
        enable_rag : bool
            Whether to enable RAG explanations
        """
        self.device = device
        self.enable_rag = enable_rag
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Load Hetionet
        logger.info("Loading Hetionet data...")
        self.loader = HetionetLoader(data_dir=data_dir)
        self.loader.load()
        self.metadata = self.loader.get_metadata()
        
        logger.info(f"Loaded graph with {self.metadata['total_nodes']} nodes")
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = self._load_model(model_path)
        
        # Initialize RAG (optional)
        self.retriever = None
        self.explanation_generator = None
        if enable_rag:
            try:
                self.retriever = RAGRetriever()
                self.explanation_generator = ExplanationGenerator(self.retriever)
                logger.info("RAG system initialized")
            except Exception as e:
                logger.warning(f"RAG system not available: {e}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config not found at {config_path}, using defaults")
            return {
                'model': {
                    'encoder': {'hidden_dim': 128, 'num_layers': 2}
                }
            }
    
    def _load_model(self, model_path: str) -> DrugRepurposingModel:
        """Load trained model from checkpoint."""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Initialize model
        num_nodes_dict = {k: v for k, v in self.metadata['node_types'].items()}
        
        model = DrugRepurposingModel(
            num_nodes_dict=num_nodes_dict,
            num_relations=self.metadata['num_edge_types'],
            hidden_dim=int(self.config['model']['encoder']['hidden_dim']),
            num_layers=int(self.config['model']['encoder']['num_layers'])
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        logger.info("Model loaded successfully")
        return model
    
    def predict(
        self,
        drug_id: Optional[str] = None,
        disease_id: Optional[str] = None,
        drug_name: Optional[str] = None,
        disease_name: Optional[str] = None,
        generate_explanation: bool = True
    ) -> PredictionResult:
        """
        Predict drug repurposing potential.
        
        Parameters
        ----------
        drug_id : Optional[str]
            Drug identifier (e.g., "DB00945")
        disease_id : Optional[str]
            Disease identifier (e.g., "DOID:1612")
        drug_name : Optional[str]
            Drug name (alternative to drug_id)
        disease_name : Optional[str]
            Disease name (alternative to disease_id)
        generate_explanation : bool
            Whether to generate explanation
        
        Returns
        -------
        PredictionResult
            Prediction result with score and explanation
        """
        # Resolve IDs from names if needed
        if drug_id is None and drug_name is not None:
            drug_id, drug_name = self._find_compound_by_name(drug_name)
        elif drug_id is not None:
            drug_name = self._get_compound_name(drug_id)
        else:
            raise ValueError("Must provide either drug_id or drug_name")
        
        if disease_id is None and disease_name is not None:
            disease_id, disease_name = self._find_disease_by_name(disease_name)
        elif disease_id is not None:
            disease_name = self._get_disease_name(disease_id)
        else:
            raise ValueError("Must provide either disease_id or disease_name")
        
        # Get node indices
        compound_idx = self.loader.node_mappings['Compound'][drug_id]
        disease_idx = self.loader.node_mappings['Disease'][disease_id]
        
        # Make prediction
        with torch.no_grad():
            compound_tensor = torch.tensor([compound_idx], device=self.device)
            disease_tensor = torch.tensor([disease_idx], device=self.device)
            
            logit = self.model(compound_tensor, disease_tensor)
            score = torch.sigmoid(logit).item()
        
        # Determine confidence
        if score > 0.7:
            confidence = "high"
        elif score > 0.4:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Generate explanation
        explanation = None
        if generate_explanation and self.explanation_generator and score > 0.3:
            try:
                explanation = self.explanation_generator.generate_explanation(
                    drug_name=drug_name,
                    disease_name=disease_name,
                    prediction_score=score
                )
            except Exception as e:
                logger.warning(f"Failed to generate explanation: {e}")
        
        return PredictionResult(
            drug_id=drug_id,
            drug_name=drug_name,
            disease_id=disease_id,
            disease_name=disease_name,
            prediction_score=score,
            confidence=confidence,
            explanation=explanation
        )
    
    def _find_compound_by_name(self, name: str) -> tuple:
        """Find compound by name."""
        compounds = self.loader.nodes_by_type.get('Compound', [])
        for compound in compounds:
            if compound['name'].lower() == name.lower():
                return compound['id'], compound['name']
        
        raise ValueError(f"Compound not found: {name}")
    
    def _find_disease_by_name(self, name: str) -> tuple:
        """Find disease by name."""
        diseases = self.loader.nodes_by_type.get('Disease', [])
        for disease in diseases:
            if disease['name'].lower() == name.lower():
                return disease['id'], disease['name']
        
        raise ValueError(f"Disease not found: {name}")
    
    def _get_compound_name(self, drug_id: str) -> str:
        """Get compound name from ID."""
        compounds = self.loader.nodes_by_type.get('Compound', [])
        for compound in compounds:
            if compound['id'] == drug_id:
                return compound['name']
        return drug_id
    
    def _get_disease_name(self, disease_id: str) -> str:
        """Get disease name from ID."""
        diseases = self.loader.nodes_by_type.get('Disease', [])
        for disease in diseases:
            if disease['id'] == disease_id:
                return disease['name']
        return disease_id
    
    def predict_top_k(
        self,
        drug_id: Optional[str] = None,
        drug_name: Optional[str] = None,
        top_k: int = 10
    ) -> list:
        """
        Find top-k diseases for a given drug.
        
        Parameters
        ----------
        drug_id : Optional[str]
            Drug identifier
        drug_name : Optional[str]
            Drug name
        top_k : int
            Number of top predictions to return
        
        Returns
        -------
        list
            List of PredictionResult objects
        """
        # Resolve drug ID
        if drug_id is None and drug_name is not None:
            drug_id, drug_name = self._find_compound_by_name(drug_name)
        elif drug_id is not None:
            drug_name = self._get_compound_name(drug_id)
        else:
            raise ValueError("Must provide either drug_id or drug_name")
        
        compound_idx = self.loader.node_mappings['Compound'][drug_id]
        diseases = self.loader.nodes_by_type.get('Disease', [])
        
        # Predict for all diseases
        predictions = []

        with torch.no_grad():
            compound_tensor = torch.tensor([compound_idx], device=self.device)

            # Predict for all diseases
            for disease in diseases:
                disease_id = disease['id']
                disease_name = disease['name']
                disease_idx = self.loader.node_mappings['Disease'][disease_id]

                disease_tensor = torch.tensor([disease_idx], device=self.device)
                logit = self.model(compound_tensor, disease_tensor)
                score = torch.sigmoid(logit).item()

                predictions.append({
                    'disease_id': disease_id,
                    'disease_name': disease_name,
                    'score': score
                })
        
        # Sort and get top-k
        predictions.sort(key=lambda x: x['score'], reverse=True)
        
        results = []
        for pred in predictions[:top_k]:
            confidence = "high" if pred['score'] > 0.7 else "medium" if pred['score'] > 0.4 else "low"
            results.append(PredictionResult(
                drug_id=drug_id,
                drug_name=drug_name,
                disease_id=pred['disease_id'],
                disease_name=pred['disease_name'],
                prediction_score=pred['score'],
                confidence=confidence
            ))
        
        return results
