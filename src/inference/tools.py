"""
LLM Tool Definitions for Drug Repurposing

Provides function calling tools for LLM agents.
"""

from typing import TypedDict, List, Optional
from .predictor import DrugRepurposingPredictor


class RepurposingPrediction(TypedDict):
    """
    Type definition for drug repurposing prediction.
    
    Attributes
    ----------
    drug_id : str
        DrugBank identifier (e.g., "DB00945" for Aspirin)
    drug_name : str
        Common drug name
    disease_id : str
        Disease Ontology identifier (e.g., "DOID:1612" for breast cancer)
    disease_name : str
        Disease name
    prediction_score : float
        Model prediction score (0-1)
    confidence : str
        Confidence level (low/medium/high)
    shared_genes : List[str]
        List of genes connecting drug to disease
    shared_pathways : List[str]
        List of biological pathways
    known_similar_indications : List[str]
        List of similar known indications
    """
    drug_id: str
    drug_name: str
    disease_id: str
    disease_name: str
    prediction_score: float
    confidence: str
    shared_genes: List[str]
    shared_pathways: List[str]
    known_similar_indications: List[str]


def get_repurposing_prediction(
    drug_id: str,
    disease_id: str,
    include_pathway_analysis: bool = True,
    top_k_genes: int = 10
) -> RepurposingPrediction:
    """
    Retrieve drug repurposing prediction from the R-GCN model.
    
    This function is designed to be used as a tool by LLM agents for
    function calling. It provides drug repurposing predictions with
    biological context.
    
    Parameters
    ----------
    drug_id : str
        DrugBank identifier (e.g., "DB00945" for Aspirin)
    disease_id : str
        Disease Ontology identifier (e.g., "DOID:1612" for breast cancer)
    include_pathway_analysis : bool, optional
        If True, return shared biological pathways between drug targets
        and disease-associated genes (default: True)
    top_k_genes : int, optional
        Number of top connecting genes to return in explanation (default: 10)
    
    Returns
    -------
    RepurposingPrediction
        Dictionary containing prediction score (0-1), confidence level,
        list of shared genes connecting drug to disease, shared pathways,
        and known similar indications for context
    
    Raises
    ------
    ValueError
        If drug_id or disease_id not found in knowledge graph
    FileNotFoundError
        If trained model not found
    
    Examples
    --------
    >>> prediction = get_repurposing_prediction(
    ...     drug_id="DB00945",  # Aspirin
    ...     disease_id="DOID:1612"  # Breast cancer
    ... )
    >>> print(f"Score: {prediction['prediction_score']:.3f}")
    >>> print(f"Confidence: {prediction['confidence']}")
    """
    # Initialize predictor
    predictor = DrugRepurposingPredictor(enable_rag=False)
    
    # Get prediction
    result = predictor.predict(
        drug_id=drug_id,
        disease_id=disease_id,
        generate_explanation=False
    )
    
    # Get shared genes and pathways (placeholder - would need graph analysis)
    shared_genes = _get_shared_genes(
        predictor, 
        drug_id, 
        disease_id, 
        top_k=top_k_genes
    )
    
    shared_pathways = []
    if include_pathway_analysis:
        shared_pathways = _get_shared_pathways(predictor, drug_id, disease_id)
    
    # Get similar indications
    similar_indications = _get_similar_indications(predictor, drug_id)
    
    return RepurposingPrediction(
        drug_id=result.drug_id,
        drug_name=result.drug_name,
        disease_id=result.disease_id,
        disease_name=result.disease_name,
        prediction_score=result.prediction_score,
        confidence=result.confidence,
        shared_genes=shared_genes,
        shared_pathways=shared_pathways,
        known_similar_indications=similar_indications
    )


def _get_shared_genes(
    predictor: DrugRepurposingPredictor,
    drug_id: str,
    disease_id: str,
    top_k: int = 10
) -> List[str]:
    """
    Find genes that connect drug to disease.
    
    This is a simplified implementation. A full implementation would:
    1. Find genes targeted by the drug (Compound-targets-Gene edges)
    2. Find genes associated with the disease (Disease-associates-Gene edges)
    3. Return the intersection
    """
    # Placeholder - would need graph traversal
    return [
        f"Gene_{i}" for i in range(min(top_k, 5))
    ]


def _get_shared_pathways(
    predictor: DrugRepurposingPredictor,
    drug_id: str,
    disease_id: str
) -> List[str]:
    """
    Find biological pathways connecting drug to disease.
    
    This would trace paths through the graph:
    Drug -> Gene -> Pathway <- Gene <- Disease
    """
    # Placeholder
    return [
        "Pathway_1",
        "Pathway_2"
    ]


def _get_similar_indications(
    predictor: DrugRepurposingPredictor,
    drug_id: str
) -> List[str]:
    """
    Get known indications for the drug.
    
    Returns list of diseases the drug is known to treat.
    """
    # Get known treats relationships
    known_pairs = predictor.loader.get_drug_disease_pairs()
    drug_pairs = known_pairs[known_pairs['drug_id'] == drug_id]
    
    return drug_pairs['disease_name'].tolist()[:5]


# Tool schema for LLM function calling
TOOL_SCHEMA = {
    "name": "get_repurposing_prediction",
    "description": "Get drug repurposing prediction for a drug-disease pair from a graph neural network model",
    "parameters": {
        "type": "object",
        "properties": {
            "drug_id": {
                "type": "string",
                "description": "DrugBank identifier (e.g., DB00945 for Aspirin)"
            },
            "disease_id": {
                "type": "string",
                "description": "Disease Ontology identifier (e.g., DOID:1612 for breast cancer)"
            },
            "include_pathway_analysis": {
                "type": "boolean",
                "description": "Whether to include shared biological pathways",
                "default": True
            },
            "top_k_genes": {
                "type": "integer",
                "description": "Number of connecting genes to return",
                "default": 10
            }
        },
        "required": ["drug_id", "disease_id"]
    }
}
