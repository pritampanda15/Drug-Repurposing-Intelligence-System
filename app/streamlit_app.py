#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Drug Repurposing Intelligence System - Streamlit Interface

A web interface for exploring drug repurposing predictions and explanations.
"""

import streamlit as st
import sys
from pathlib import Path
import torch
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.hetionet_loader import HetionetLoader
from src.models.link_predictor import DrugRepurposingModel
from src.rag.retriever import RAGRetriever
from src.rag.explanation_generator import ExplanationGenerator


# Page config
st.set_page_config(
    page_title="Drug Repurposing Intelligence",
    page_icon="💊",
    layout="wide"
)

# Title
st.title("💊 Drug Repurposing Intelligence System")
st.markdown("""
Discover novel drug-disease relationships using graph neural networks and AI-powered explanations.
""")

# Sidebar
st.sidebar.header("Configuration")

model_path = st.sidebar.text_input(
    "Model Path",
    value="models/checkpoints/best_model.pt"
)

data_dir = st.sidebar.text_input(
    "Hetionet Data Directory",
    value="data/raw/hetionet"
)

use_rag = st.sidebar.checkbox("Enable RAG Explanations", value=True)


@st.cache_resource
def load_hetionet(data_dir):
    """Load Hetionet data."""
    loader = HetionetLoader(data_dir=data_dir)
    loader.load()
    return loader


@st.cache_resource
def load_model(model_path, _loader):
    """Load trained model."""
    if not Path(model_path).exists():
        return None

    metadata = _loader.get_metadata()
    num_nodes_dict = {k: v for k, v in metadata['node_types'].items()}
    
    model = DrugRepurposingModel(
        num_nodes_dict=num_nodes_dict,
        num_relations=metadata['num_edge_types'],
        hidden_dim=128,
        num_layers=2
    )
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


@st.cache_resource
def load_rag_system():
    """Load RAG retriever and generator."""
    try:
        retriever = RAGRetriever()
        generator = ExplanationGenerator(retriever)
        return retriever, generator
    except Exception as e:
        st.warning(f"RAG system not available: {e}")
        return None, None


# Main content
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Drug-Disease Prediction",
    "📊 Knowledge Graph Explorer",
    "📈 Statistics",
    "ℹ️ About"
])

# Tab 1: Prediction
with tab1:
    st.header("Drug-Disease Prediction")
    
    try:
        loader = load_hetionet(data_dir)
        st.success(f"✓ Loaded Hetionet with {loader.get_metadata()['total_nodes']:,} nodes")
        
        # Get available drugs and diseases
        compounds = loader.nodes_by_type.get('Compound', [])
        diseases = loader.nodes_by_type.get('Disease', [])

        # Sort alphabetically for easier searching
        compounds_sorted = sorted(compounds, key=lambda x: x['name'])
        diseases_sorted = sorted(diseases, key=lambda x: x['name'])

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Select Drug")
            st.caption(f"📊 {len(compounds_sorted):,} compounds available")

            # All drugs - selectbox has built-in search
            drug_names = [c['name'] for c in compounds_sorted]
            selected_drug = st.selectbox(
                "Drug (start typing to search)",
                options=drug_names,
                help="Select a compound from Hetionet. Start typing to search.",
                key="drug_select"
            )

            if selected_drug:
                drug_idx = next(i for i, c in enumerate(compounds) if c['name'] == selected_drug)
                drug_info = compounds[drug_idx]
                st.info(f"**ID:** {drug_info['id']}")

        with col2:
            st.subheader("Select Disease")
            st.caption(f"📊 {len(diseases_sorted):,} diseases available")

            # All diseases - selectbox has built-in search
            disease_names = [d['name'] for d in diseases_sorted]
            selected_disease = st.selectbox(
                "Disease (start typing to search)",
                options=disease_names,
                help="Select a disease from Hetionet. Start typing to search.",
                key="disease_select"
            )

            if selected_disease:
                disease_idx = next(i for i, d in enumerate(diseases) if d['name'] == selected_disease)
                disease_info = diseases[disease_idx]
                st.info(f"**ID:** {disease_info['id']}")
        
        if st.button("🔮 Predict Repurposing Potential", type="primary"):
            with st.spinner("Running prediction..."):
                # Load model
                model = load_model(model_path, loader)
                
                if model is None:
                    st.error(f"Model not found at {model_path}. Please train the model first.")
                else:
                    # Get indices
                    drug_idx_tensor = torch.tensor([drug_idx])
                    disease_idx_tensor = torch.tensor([disease_idx])
                    
                    # Predict
                    with torch.no_grad():
                        logit = model(drug_idx_tensor, disease_idx_tensor)
                        score = torch.sigmoid(logit).item()
                    
                    # Display results
                    st.divider()
                    st.subheader("Prediction Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Prediction Score", f"{score:.3f}")
                    
                    with col2:
                        confidence = "High" if score > 0.7 else "Medium" if score > 0.4 else "Low"
                        st.metric("Confidence", confidence)
                    
                    with col3:
                        recommendation = "✓ Promising" if score > 0.6 else "⚠ Uncertain"
                        st.metric("Recommendation", recommendation)
                    
                    # Progress bar for score
                    st.progress(score, text=f"Repurposing Potential: {score*100:.1f}%")
                    
                    # RAG Explanation
                    if use_rag:
                        st.divider()
                        st.subheader("AI-Generated Explanation")

                        retriever, generator = load_rag_system()

                        if generator:
                            with st.spinner("Generating explanation..."):
                                try:
                                    explanation = generator.generate_explanation(
                                        drug_name=selected_drug,
                                        disease_name=selected_disease,
                                        prediction_score=score
                                    )
                                    st.markdown(explanation)
                                except Exception as e:
                                    st.error(f"Failed to generate explanation: {e}")
                                    st.info("""
                                    **Possible issues:**
                                    1. OpenAI API key not set or invalid
                                    2. No documents indexed in ChromaDB
                                    3. Rate limit exceeded

                                    **To fix:**
                                    - Set `OPENAI_API_KEY` in `.env` file
                                    - Run `python3 scripts/index_documents.py`
                                    """)
                        else:
                            st.warning("""
                            **Explanation unavailable**: RAG system not configured.

                            To enable explanations:
                            1. Run `python3 scripts/index_documents.py` to index documents
                            2. Set `OPENAI_API_KEY` in `.env` file
                            3. Restart the Streamlit app

                            Even without RAG, you still get the prediction score above!
                            """)
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Make sure you've run `python scripts/download_data.py` first.")

# Tab 2: Knowledge Graph Explorer
with tab2:
    st.header("Knowledge Graph Explorer")
    
    try:
        loader = load_hetionet(data_dir)
        metadata = loader.get_metadata()
        
        st.subheader("Node Type Distribution")
        
        node_df = pd.DataFrame([
            {"Type": k, "Count": v}
            for k, v in metadata['node_types'].items()
        ]).sort_values("Count", ascending=False)
        
        fig = px.bar(
            node_df,
            x="Type",
            y="Count",
            title="Number of Nodes by Type",
            color="Count",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Edge Type Distribution")
        
        edge_df = pd.DataFrame([
            {"Type": k, "Count": v}
            for k, v in metadata['edge_types'].items()
        ]).sort_values("Count", ascending=False).head(20)
        
        fig = px.bar(
            edge_df,
            x="Count",
            y="Type",
            orientation='h',
            title="Top 20 Edge Types by Count",
            color="Count",
            color_continuous_scale="Greens"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error: {e}")

# Tab 3: Statistics
with tab3:
    st.header("System Statistics")
    
    try:
        loader = load_hetionet(data_dir)
        metadata = loader.get_metadata()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Nodes", f"{metadata['total_nodes']:,}")
        
        with col2:
            st.metric("Total Edges", f"{metadata['total_edges']:,}")
        
        with col3:
            st.metric("Node Types", metadata['num_node_types'])
        
        st.divider()
        
        # Drug-Disease pairs
        drug_disease_pairs = loader.get_drug_disease_pairs()
        st.subheader("Drug-Disease Relationships")
        st.metric("Known Treatment Relationships", len(drug_disease_pairs))
        
        st.dataframe(
            drug_disease_pairs.head(10),
            use_container_width=True
        )
        
    except Exception as e:
        st.error(f"Error: {e}")

# Tab 4: About
with tab4:
    st.header("About This System")
    
    st.markdown("""
    ## Drug Repurposing Intelligence System
    
    This application combines **graph neural networks** with **retrieval-augmented generation (RAG)** 
    to predict and explain drug repurposing opportunities.
    
    ### How It Works
    
    1. **Knowledge Graph**: Built from Hetionet, a biomedical knowledge graph with 47K+ nodes and 2.25M+ edges
    2. **Graph Neural Network**: R-GCN model learns to predict drug-disease relationships
    3. **RAG System**: Retrieves relevant scientific context and generates explanations using LLMs
    
    ### Features
    
    - 🔍 **Drug-Disease Prediction**: Predict repurposing potential for any drug-disease pair
    - 💡 **AI Explanations**: Get detailed biological explanations for predictions
    - 📊 **Graph Exploration**: Explore the biomedical knowledge graph
    - 📈 **Statistics**: View system statistics and known relationships
    
    ### Data Sources
    
    - **Hetionet**: Biomedical knowledge graph (Himmelstein et al., 2017)
    - **DrugBank**: Drug mechanisms and indications
    - **PubMed**: Scientific literature
    
    ### Setup
    
    To use this system:
    
    ```bash
    # 1. Download data
    python scripts/download_data.py
    
    # 2. Build knowledge graph
    python scripts/build_knowledge_graph.py
    
    # 3. Train model
    python scripts/train_model.py --epochs 50
    
    # 4. Index documents for RAG
    python scripts/index_documents.py
    
    # 5. Run this app
    streamlit run app/streamlit_app.py
    ```
    
    ### Citation
    
    If you use this system in your research, please cite:
    
    - Himmelstein et al. (2017). "Systematic integration of biomedical knowledge prioritizes drugs for repurposing." *eLife*
    - Schlichtkrull et al. (2018). "Modeling Relational Data with Graph Convolutional Networks." *ESWC*
    
    ### Links
    
    - [GitHub Repository](#)
    - [Documentation](QUICKSTART.md)
    - [Hetionet](https://het.io)
    """)
    
    st.divider()
    
    st.info("""
    **Note**: This is a research tool. Predictions should be validated through proper 
    experimental and clinical studies before any therapeutic use.
    """)

# Footer
st.sidebar.divider()
st.sidebar.markdown("---")
st.sidebar.caption("Drug Repurposing Intelligence System v0.1.0")
