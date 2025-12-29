"""Example usage of document segmentation with both OpenAI and Ollama backends."""

from document_segmenter import (
    PdfChunker,
    OllamaPdfChunker,
    DocumentSegmenter,
    OllamaDocumentSegmenter,
    Breakpoints
)


def example_openai_segmentation():
    """Example: Using OpenAI-based document segmentation."""
    print("=== OpenAI Document Segmentation Example ===\n")
    
    # Initialize the OpenAI-based chunker
    chunker = PdfChunker()
    
    # Process a PDF file
    pdf_path = "path/to/your/document.pdf"
    try:
        chunks = chunker.chunk_pdf(pdf_path)
        
        print(f"Successfully segmented document into {len(chunks)} chunks\n")
        
        for i, chunk in enumerate(chunks, 1):
            print(f"Chunk {i}:")
            print(f"  Words: {len(chunk.split())}")
            print(f"  Preview: {chunk[:100]}...\n")
    except Exception as e:
        print(f"Error: {e}")


def example_ollama_segmentation():
    """Example: Using Ollama-based document segmentation."""
    print("=== Ollama Document Segmentation Example ===\n")
    
    # Initialize the Ollama-based chunker
    chunker = OllamaPdfChunker()
    
    # Process a PDF file
    pdf_path = "path/to/your/document.pdf"
    try:
        chunks = chunker.chunk_pdf(pdf_path)
        
        print(f"Successfully segmented document into {len(chunks)} chunks\n")
        
        for i, chunk in enumerate(chunks, 1):
            print(f"Chunk {i}:")
            print(f"  Words: {len(chunk.split())}")
            print(f"  Preview: {chunk[:100]}...\n")
    except Exception as e:
        print(f"Error: {e}")


def example_direct_segmenter_usage():
    """Example: Using document segmenters directly with text."""
    print("=== Direct Segmenter Usage Example ===\n")
    
    # Sample sentences
    sentences = [
        "Introduction to machine learning.",
        "Machine learning is a subset of artificial intelligence.",
        "It focuses on enabling computers to learn from data.",
        "Deep learning is a specialized branch of machine learning.",
        "It uses neural networks with multiple layers.",
        "Now let's discuss data preprocessing.",
        "Data preprocessing is crucial for model performance.",
        "It includes cleaning, normalization, and feature engineering."
    ]
    
    # Using OpenAI segmenter
    print("Using OpenAI Segmenter:")
    try:
        openai_segmenter = DocumentSegmenter()
        breakpoints: Breakpoints = openai_segmenter.determine_breakpoints(sentences)
        
        print(f"  Breakpoints: {breakpoints.breakpoints}")
        print(f"  Notes: {breakpoints.notes}\n")
    except Exception as e:
        print(f"  Error: {e}\n")
    
    # Using Ollama segmenter
    print("Using Ollama Segmenter:")
    try:
        ollama_segmenter = OllamaDocumentSegmenter()
        breakpoints: Breakpoints = ollama_segmenter.determine_breakpoints(sentences)
        
        print(f"  Breakpoints: {breakpoints.breakpoints}")
        print(f"  Notes: {breakpoints.notes}\n")
    except Exception as e:
        print(f"  Error: {e}\n")


if __name__ == "__main__":
    print("Document Segmentation Engine - Usage Examples\n")
    print("=" * 60)
    print()
    
    # Choose which example to run
    print("Available examples:")
    print("1. OpenAI-based PDF segmentation")
    print("2. Ollama-based PDF segmentation")
    print("3. Direct segmenter usage (no PDF)")
    print()
    
    choice = input("Enter your choice (1-3): ").strip()
    print()
    
    if choice == "1":
        example_openai_segmentation()
    elif choice == "2":
        example_ollama_segmentation()
    elif choice == "3":
        example_direct_segmenter_usage()
    else:
        print("Invalid choice. Running all examples:\n")
        example_direct_segmenter_usage()
