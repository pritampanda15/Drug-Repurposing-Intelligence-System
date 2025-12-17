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
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

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
    try:
        retriever = RAGRetriever(
            persist_directory="data/knowledge_base/chroma"
        )
        generator = ExplanationGenerator(retriever)
        return retriever, generator
    except Exception as e:
        st.warning(f"RAG system not available: {e}")
        return None, None




# Main content
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔍 Drug-Disease Prediction",
    "💬 Drug Q&A",
    "📄 Paper Analysis",
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

# Tab 2: Drug Q&A
with tab2:
    st.header("💬 Ask Questions About Drugs")
    st.markdown("""
    Ask any question about a specific drug's mechanism, indications, or pharmacology.
    The system will search the knowledge base and provide AI-powered answers.
    """)

    try:
        loader = load_hetionet(data_dir)
        compounds = loader.nodes_by_type.get('Compound', [])
        compounds_sorted = sorted(compounds, key=lambda x: x['name'])

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Select Drug")
            drug_names = [c['name'] for c in compounds_sorted]
            qa_drug = st.selectbox(
                "Drug",
                options=drug_names,
                help="Select a drug to ask questions about",
                key="qa_drug_select"
            )

            if qa_drug:
                drug_idx = next(i for i, c in enumerate(compounds) if c['name'] == qa_drug)
                drug_info = compounds[drug_idx]
                st.info(f"**ID:** {drug_info['id']}")

                st.markdown("**Example Questions:**")
                st.markdown("- What is the mechanism of action?")
                st.markdown("- What diseases does it treat?")
                st.markdown("- What are the side effects?")
                st.markdown("- How does it work?")

        with col2:
            st.subheader("Your Question")

            # Question input
            user_question = st.text_input(
                "Ask a question about this drug:",
                placeholder=f"e.g., How does {qa_drug if qa_drug else 'this drug'} work?",
                key="qa_question"
            )

            if st.button("🔍 Get Answer", key="qa_button"):
                if not user_question:
                    st.warning("Please enter a question first.")
                else:
                    with st.spinner("Searching knowledge base and generating answer..."):
                        retriever, generator = load_rag_system()

                        if retriever and generator:
                            try:
                                # Search for drug-specific information
                                search_query = f"{qa_drug} {user_question}"
                                docs = retriever.retrieve(
                                    query=search_query,
                                    collection_name="drug_mechanisms",
                                    top_k=3
                                )

                                # Build context from retrieved docs
                                context = "\n\n".join([
                                    f"Source {i+1}:\n{doc['text'][:500]}"
                                    for i, doc in enumerate(docs)
                                ])

                                # Generate answer
                                if not generator.client:
                                    st.error("OpenAI API key not configured. Cannot generate answer.")
                                    st.info("Retrieved information:")
                                    for i, doc in enumerate(docs):
                                        with st.expander(f"Source {i+1}"):
                                            st.write(doc['text'][:500])
                                else:
                                    prompt = f"""Answer this question about {qa_drug}:

Question: {user_question}

Relevant Information:
{context}

Provide a clear, concise answer based on the information above. If the information doesn't fully answer the question, say so."""

                                    response = generator.client.chat.completions.create(
                                        model=generator.model,
                                        messages=[
                                            {"role": "system", "content": "You are a knowledgeable pharmacology assistant. Provide accurate, evidence-based answers about drugs."},
                                            {"role": "user", "content": prompt}
                                        ],
                                        temperature=0.3,
                                        max_tokens=500
                                    )

                                    answer = response.choices[0].message.content

                                    # Display answer
                                    st.success("**Answer:**")
                                    st.markdown(answer)

                                    # Show sources
                                    with st.expander("📚 View Sources"):
                                        for i, doc in enumerate(docs):
                                            st.markdown(f"**Source {i+1}:**")
                                            st.text(doc['text'][:300] + "...")
                                            st.markdown("---")

                            except Exception as e:
                                st.error(f"Error generating answer: {e}")
                                st.info("""
                                **Possible issues:**
                                - OpenAI API key not set or invalid
                                - No documents indexed
                                - Rate limit exceeded

                                **To fix:**
                                - Set `OPENAI_API_KEY` in `.env` file
                                - Run `python3 scripts/index_documents.py`
                                - Restart Streamlit
                                """)
                        else:
                            st.warning("""
                            **Q&A system not available.**

                            To enable Q&A:
                            1. Index documents: `python3 scripts/index_documents.py`
                            2. Set `OPENAI_API_KEY` in `.env` file
                            3. Restart Streamlit app
                            """)

    except Exception as e:
        st.error(f"Error: {e}")

# Tab 3: Paper Analysis
with tab3:
    st.header("📄 Scientific Paper Analysis")
    st.markdown("""
    Upload a scientific paper (PDF) to extract chemical compounds and generate an AI-powered summary.
    """)

    uploaded_file = st.file_uploader("Upload PDF Paper", type=['pdf'])

    if uploaded_file is not None:
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        st.success(f"✓ Uploaded: {uploaded_file.name}")

        # Extract text
        with st.spinner("Extracting text from PDF..."):
            try:
                from src.paper_analysis.pdf_extractor import PDFExtractor

                extractor = PDFExtractor()
                text = extractor.extract_text(tmp_path)
                metadata = extractor.extract_metadata(tmp_path)

                st.info(f"📄 Pages: {metadata.get('num_pages', 'Unknown')}")

                # Show text preview
                with st.expander("📝 View Extracted Text (preview)"):
                    st.text(text[:2000] + "..." if len(text) > 2000 else text)

            except Exception as e:
                st.error(f"Failed to extract text: {e}")
                text = None

        if text:
            col1, col2 = st.columns(2)

            # Extract chemicals
            with col1:
                st.subheader("🧪 Extracted Chemicals")

                with st.spinner("Identifying chemicals..."):
                    try:
                        from src.paper_analysis.chemical_extractor import ChemicalExtractor

                        chem_extractor = ChemicalExtractor()
                        chemicals = chem_extractor.extract_chemicals(
                            text,
                            min_length=4,
                            max_results=20,
                            validate=False  # Set to True for PubChem validation (slower)
                        )

                        if chemicals:
                            st.success(f"Found {len(chemicals)} potential chemicals")

                            # Display as table
                            import pandas as pd
                            chem_df = pd.DataFrame(chemicals)
                            st.dataframe(
                                chem_df[['name', 'count']].head(15),
                                width='stretch'
                            )

                            # Select chemical for visualization
                            selected_chem = st.selectbox(
                                "Select chemical to visualize:",
                                options=[c['name'] for c in chemicals[:10]]
                            )

                            if st.button("🎨 Show 2D Structure"):
                                with st.spinner(f"Generating structure for {selected_chem}..."):
                                    try:
                                        from src.paper_analysis.structure_visualizer import StructureVisualizer
                                        from io import BytesIO
                                        from PIL import Image

                                        visualizer = StructureVisualizer()

                                        # Get SMILES first
                                        smiles = visualizer.name_to_smiles(selected_chem)

                                        if not smiles:
                                            st.error(f"❌ Could not find '{selected_chem}' in PubChem database.")
                                            st.info("Try using the exact chemical name or select another compound.")
                                        else:
                                            st.success(f"✓ Found: **{selected_chem}**")
                                            st.code(f"SMILES: {smiles}", language=None)

                                            # Generate image
                                            img_bytes = visualizer.draw_from_smiles(smiles)

                                            if img_bytes:
                                                # Convert bytes to PIL Image for Streamlit
                                                img = Image.open(BytesIO(img_bytes))
                                                st.image(img, caption=f"2D Structure of {selected_chem}", width='stretch')

                                                # Show properties
                                                props = visualizer.get_molecular_properties(smiles)
                                                if props:
                                                    st.markdown("**Molecular Properties:**")
                                                    prop_col1, prop_col2, prop_col3 = st.columns(3)
                                                    with prop_col1:
                                                        st.metric("Molecular Weight", f"{props.get('molecular_weight', 0):.2f}")
                                                        st.metric("H-Bond Donors", props.get('h_bond_donors', 0))
                                                    with prop_col2:
                                                        st.metric("LogP", f"{props.get('logp', 0):.2f}")
                                                        st.metric("H-Bond Acceptors", props.get('h_bond_acceptors', 0))
                                                    with prop_col3:
                                                        st.metric("Rotatable Bonds", props.get('rotatable_bonds', 0))
                                                        st.metric("TPSA", f"{props.get('tpsa', 0):.2f} Ų")
                                            else:
                                                st.error("Failed to generate molecular structure image.")

                                    except Exception as e:
                                        st.error(f"Structure visualization failed: {e}")
                                        import traceback
                                        with st.expander("Show error details"):
                                            st.code(traceback.format_exc())
                        else:
                            st.warning("No chemicals found in the paper.")

                    except Exception as e:
                        st.error(f"Chemical extraction failed: {e}")

            # Generate summary
            with col2:
                st.subheader("📋 AI-Generated Summary")

                if st.button("✨ Generate Summary", type="primary"):
                    with st.spinner("Analyzing paper and generating summary..."):
                        try:
                            from src.paper_analysis.summarizer import PaperSummarizer

                            summarizer = PaperSummarizer()
                            summary = summarizer.summarize_paper(text, max_length=400)

                            if summary:
                                # Display structured summary
                                if summary.get('objective'):
                                    st.markdown("**🎯 Main Objective:**")
                                    st.write(summary['objective'])

                                if summary.get('methods'):
                                    st.markdown("**🔬 Key Methods:**")
                                    st.write(summary['methods'])

                                if summary.get('findings'):
                                    st.markdown("**🔍 Major Findings:**")
                                    st.write(summary['findings'])

                                if summary.get('relevance'):
                                    st.markdown("**💊 Clinical Relevance:**")
                                    st.write(summary['relevance'])

                                if summary.get('chemicals'):
                                    st.markdown("**🧪 Key Chemicals:**")
                                    st.write(summary['chemicals'])

                                # Full summary in expander
                                with st.expander("📄 Full Summary"):
                                    st.write(summary.get('full_summary', ''))
                            else:
                                st.error("Failed to generate summary")

                        except Exception as e:
                            st.error(f"Summarization failed: {e}")
                            st.info("Make sure OPENAI_API_KEY is set in your .env file")

        # Cleanup temp file
        import os
        try:
            os.unlink(tmp_path)
        except:
            pass

# Tab 4: Knowledge Graph Explorer
with tab4:
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
        st.plotly_chart(fig, width='stretch')

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

# Tab 5: Statistics
with tab5:
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

# Tab 6: About
with tab6:
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
