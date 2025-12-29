"""Streamlit application for PDF document segmentation and embedding generation."""

import streamlit as st
from document_segmenter import PdfChunker, OllamaPdfChunker, TitanEmbeddingGenerator
import os
import tempfile


def main():
    """Main Streamlit application for document segmentation."""
    st.set_page_config(
        page_title="Document Segmenting Engine",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Document Segmenting Engine")
    st.markdown("""
    Upload a PDF document to:
    1. Extract and segment text using semantic chunking
    2. Generate embeddings using AWS Titan
    3. Store vectors in AWS S3 Vectors
    """)
    
    # Inference backend selection
    st.subheader("⚙️ Inference Configuration")
    col_config1, col_config2 = st.columns([2, 3])
    
    with col_config1:
        inference_backend = st.radio(
            "Select Inference Backend",
            options=["Online (OpenAI)", "Local (Ollama)"],
            help="Choose between cloud-based OpenAI or local Ollama models"
        )
    
    with col_config2:
        if inference_backend == "Online (OpenAI)":
            st.info("""
            🌐 **Online Mode**
            - Uses OpenAI API
            - Requires API key
            - High quality results
            - API costs apply
            """)
        else:
            st.info("""
            🏠 **Local Mode**
            - Uses Ollama
            - Runs on your machine
            - No API costs
            - Privacy-preserving
            - Requires Ollama installed
            """)
    
    st.divider()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF document to process"
    )
    
    if uploaded_file is not None:
        # Display file details
        st.info(f"📁 File: {uploaded_file.name} ({uploaded_file.size / 1024:.2f} KB)")
        
        # Create two columns for buttons
        col1, col2 = st.columns(2)
        
        with col1:
            chunk_button = st.button("🔍 Segment Document", use_container_width=True)
        
        with col2:
            vectorize_button = st.button("🚀 Segment & Vectorize", use_container_width=True)
        
        # Process: Segmentation only
        if chunk_button:
            backend_name = "OpenAI" if inference_backend == "Online (OpenAI)" else "Ollama"
            with st.spinner(f"Segmenting document using {backend_name}..."):
                try:
                    # Save uploaded file to temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # Process the PDF with selected backend
                    if inference_backend == "Online (OpenAI)":
                        chunker = PdfChunker()
                    else:
                        chunker = OllamaPdfChunker()
                    
                    chunks = chunker.chunk_pdf(tmp_file_path)
                    
                    # Clean up temporary file
                    os.unlink(tmp_file_path)
                    
                    # Display results
                    st.success(f"✅ Successfully segmented document into {len(chunks)} chunks using {backend_name}")
                    
                    # Show chunks in expandable sections
                    st.subheader("📝 Generated Chunks")
                    for i, chunk in enumerate(chunks, 1):
                        with st.expander(f"Chunk {i} ({len(chunk.split())} words)"):
                            st.write(chunk)
                    
                    # Store chunks in session state for potential vectorization
                    st.session_state['chunks'] = chunks
                    
                except Exception as e:
                    st.error(f"❌ Error during segmentation: {str(e)}")
        
        # Process: Segmentation + Vectorization
        if vectorize_button:
            backend_name = "OpenAI" if inference_backend == "Online (OpenAI)" else "Ollama"
            with st.spinner(f"Segmenting and vectorizing document using {backend_name}..."):
                try:
                    # Save uploaded file to temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # Process the PDF with selected backend
                    if inference_backend == "Online (OpenAI)":
                        chunker = PdfChunker()
                    else:
                        chunker = OllamaPdfChunker()
                    
                    chunks = chunker.chunk_pdf(tmp_file_path)
                    
                    # Clean up temporary file
                    os.unlink(tmp_file_path)
                    
                    st.success(f"✅ Successfully segmented document into {len(chunks)} chunks using {backend_name}")
                    
                    # Generate and store embeddings
                    st.info("🔄 Generating embeddings and storing in S3 Vectors...")
                    generator = TitanEmbeddingGenerator()
                    generator.vectorize_and_store(chunks)
                    
                    st.success("✅ Successfully generated and stored vector embeddings!")
                    
                    # Display summary
                    st.subheader("📊 Processing Summary")
                    st.metric("Total Chunks", len(chunks))
                    st.metric("Total Vectors Stored", len(chunks))
                    
                    # Show chunks in expandable sections
                    st.subheader("📝 Generated Chunks")
                    for i, chunk in enumerate(chunks, 1):
                        with st.expander(f"Chunk {i} ({len(chunk.split())} words)"):
                            st.write(chunk)
                    
                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This application supports multiple inference backends:
        - **OpenAI** for cloud-based semantic segmentation
        - **Ollama** for local semantic segmentation
        - **AWS Bedrock Titan** for embeddings
        - **AWS S3 Vectors** for storage
        
        ### Configuration Required
        
        **For OpenAI:**
        - `OPENAI_API_KEY`
        - `MODEL_ID`
        
        **For Ollama:**
        - `OLLAMA_MODEL_ID`
        - `OLLAMA_BASE_URL` (optional)
        - Ollama server running locally
        
        **For AWS:**
        - AWS credentials (via boto3)
        """)
        
        st.header("📚 Features")
        st.markdown("""
        - ✂️ Semantic text segmentation
        - 🧠 AI-powered chunk boundary detection
        - 🌐 Cloud or local inference options
        - 🔢 Vector embedding generation
        - ☁️ Cloud storage integration
        - 🔒 Privacy-preserving local mode
        """)


if __name__ == "__main__":
    main()
