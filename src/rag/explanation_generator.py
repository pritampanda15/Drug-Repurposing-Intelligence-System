"""
LLM-based explanation generation for drug repurposing predictions.
"""

from typing import Dict, List, Optional
import logging
import os

from .retriever import RAGRetriever


logger = logging.getLogger(__name__)


class ExplanationGenerator:
    """
    Generate natural language explanations for drug repurposing predictions.
    """

    def __init__(
        self,
        retriever: RAGRetriever,
        llm_provider: str = "openai",
        model: str = "gpt-4"
    ):
        """
        Initialize explanation generator.

        Parameters
        ----------
        retriever : RAGRetriever
            RAG retriever for context
        llm_provider : str
            LLM provider (openai, anthropic)
        model : str
            Model name
        """
        self.retriever = retriever
        self.llm_provider = llm_provider
        self.model = model

        if llm_provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                logger.warning("OpenAI not installed. Install with: pip install openai")
                self.client = None

    def generate_explanation(
        self,
        drug_name: str,
        disease_name: str,
        prediction_score: float,
        context_docs: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate explanation for a prediction.

        Parameters
        ----------
        drug_name : str
            Drug name
        disease_name : str
            Disease name
        prediction_score : float
            Model prediction score (0-1)
        context_docs : Optional[List[Dict]]
            Retrieved context documents

        Returns
        -------
        str
            Generated explanation
        """
        # Retrieve context if not provided
        if context_docs is None:
            retrieved = self.retriever.retrieve_for_prediction(
                drug_name, disease_name, top_k=3
            )
            context_docs = retrieved.get('drug_mechanism', [])

        # Build context string
        context = "\n\n".join([
            f"Source: {doc['metadata'].get('source', 'unknown')}\n{doc['text']}"
            for doc in context_docs[:3]
        ])

        # Build prompt
        prompt = f"""Based on the following biomedical knowledge, explain why {drug_name} might be effective for treating {disease_name}.

Prediction Score: {prediction_score:.2f} (0-1 scale)

Relevant Scientific Context:
{context}

Provide a concise explanation covering:
1. The mechanism of action of {drug_name}
2. How this mechanism might address {disease_name}
3. Confidence assessment based on the evidence
4. Suggested validation steps

Keep the explanation scientific but accessible."""

        # Generate (placeholder if no API key)
        if self.client is None:
            return self._generate_fallback_explanation(drug_name, disease_name, prediction_score)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a biomedical AI assistant specialized in drug repurposing."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._generate_fallback_explanation(drug_name, disease_name, prediction_score)

    def _generate_fallback_explanation(
        self,
        drug_name: str,
        disease_name: str,
        score: float
    ) -> str:
        """Generate a simple fallback explanation."""
        confidence = "high" if score > 0.7 else "moderate" if score > 0.4 else "low"

        return f"""Drug Repurposing Prediction:
Drug: {drug_name}
Disease: {disease_name}
Prediction Score: {score:.3f}
Confidence: {confidence}

This prediction is based on graph neural network analysis of biomedical knowledge.
A {confidence} score suggests that {drug_name} may have {confidence} potential for treating {disease_name}.

To generate detailed explanations with biological mechanisms, configure your OpenAI API key in the .env file.

Recommended next steps:
1. Review existing literature on {drug_name}
2. Investigate shared biological pathways
3. Consider preclinical validation studies"""
