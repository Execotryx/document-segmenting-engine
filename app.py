"""Streamlit application for PDF document segmentation and embedding generation."""

import streamlit as st
from document_segmenter import PdfChunker, TitanEmbeddingGenerator
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
            with st.spinner("Segmenting document..."):
                try:
                    # Save uploaded file to temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # Process the PDF
                    chunker = PdfChunker()
                    chunks = chunker.chunk_pdf(tmp_file_path)
                    
                    # Clean up temporary file
                    os.unlink(tmp_file_path)
                    
                    # Display results
                    st.success(f"✅ Successfully segmented document into {len(chunks)} chunks")
                    
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
            with st.spinner("Segmenting and vectorizing document..."):
                try:
                    # Save uploaded file to temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # Process the PDF
                    chunker = PdfChunker()
                    chunks = chunker.chunk_pdf(tmp_file_path)
                    
                    # Clean up temporary file
                    os.unlink(tmp_file_path)
                    
                    st.success(f"✅ Successfully segmented document into {len(chunks)} chunks")
                    
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
        This application uses:
        - **OpenAI** for semantic segmentation
        - **AWS Bedrock Titan** for embeddings
        - **AWS S3 Vectors** for storage
        
        ### Configuration Required
        Set the following environment variables:
        - `OPENAI_API_KEY`
        - `MODEL_ID`
        - AWS credentials (via boto3)
        """)
        
        st.header("📚 Features")
        st.markdown("""
        - ✂️ Semantic text segmentation
        - 🧠 AI-powered chunk boundary detection
        - 🔢 Vector embedding generation
        - ☁️ Cloud storage integration
        """)


if __name__ == "__main__":
    main()
