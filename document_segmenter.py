from openai import OpenAI
from openai.types.responses import Response
from dotenv import load_dotenv, find_dotenv
from os import getenv
from pypdf import PdfReader
from pydantic import BaseModel
import nltk  # type: ignore
from ollama import ChatResponse
from ollama_ai_core import AICore
from ollama_ai_config import OllamaAIConfig
import json

#region OpenAIConfig class - configuration for OpenAI API

class OpenAIConfig:
    """Configuration manager for OpenAI API credentials and settings.
    
    Loads API key and model ID from environment variables using python-dotenv.
    Requires OPENAI_API_KEY and MODEL_ID to be set in .env file.
    
    Raises:
        ValueError: If required environment variables are not set.
    """

    @property
    def api_key(self) -> str:
        return self.__api_key

    @property
    def model_id(self) -> str:
        return self.__model_id
    
    def __init__(self) -> None:
        load_dotenv(find_dotenv(), override=True)
        self.__api_key: str = self.__get_api_key()
        self.__model_id: str = self.__get_model_id()

    def __get_api_key(self) -> str:
        return self.__try_get_value("OPENAI_API_KEY")

    def __get_model_id(self) -> str:
        return self.__try_get_value("MODEL_ID")

    def __try_get_value(self, key: str) -> str:
        value: str | None = getenv(key)
        if not value:
            raise ValueError(f"{key} environment variable is not set.")

        return value

#endregion


#region DocumentSegmenter class - handles document segmentation using OpenAI LLM, to imitate the semantic chunking of AWS Knowledge Base

class Breakpoints(BaseModel):
    """Pydantic model for document segmentation breakpoints.
    
    Attributes:
        breakpoints: List of sentence indices where new chunks should begin.
        notes: Explanatory notes about why breakpoints were placed at those locations.
    """
    breakpoints: list[int]
    notes: list[str]


class DocumentSegmenter:
    """Handles document segmentation using OpenAI LLM for semantic chunking.
    
    Implements semantic chunk boundary detection similar to AWS Knowledge Base chunking.
    Uses OpenAI's structured output API to identify logical breakpoints in document text.
    """

    @property
    def _config(self) -> OpenAIConfig:
        return self.__config

    @property
    def _system_prompt(self) -> str:
        return self.__system_prompt

    @property
    def _client(self) -> OpenAI:
        return self.__client

    def __init__(self) -> None:
        self.__config: OpenAIConfig = OpenAIConfig()
        self.__system_prompt: str = (
            "You are a Document Segmentation Engine implementing semantic chunk boundary detection.\n\n"
            "Your task is to identify where new retrieval chunks should begin.\n"
            "You DO NOT output the chunks themselves — only the sentence indices where a new chunk starts.\n\n"
            "### CORE LOGIC\n\n"
            "1. Semantic Continuity\n"
            "- Read the document sentence by sentence.\n"
            "- Maintain a running notion of the current topic or line of reasoning.\n"
            "- As long as new sentences extend or elaborate the same idea, keep them in the same chunk.\n\n"
            "2. Topic Shift Detection\n"
            "Insert a breakpoint when there is a clear semantic transition, such as:\n"
            "- A new section or heading\n"
            "- A change in subject, goal, or reasoning\n"
            "- Discourse markers like “Moving on…”, “In contrast…”, “Chapter 2”\n"
            "High cohesion markers (“Therefore…”, “Additionally…”) indicate continuation.\n\n"
            "3. Size Awareness (Approximate)\n"
            "- Target chunk size: ~200–450 tokens (approximate).\n"
            "- If a chunk is clearly becoming too large, prefer inserting a breakpoint at the nearest logical pause (sentence boundary, paragraph break, or section boundary).\n"
            "- Avoid creating chunks smaller than ~50 tokens unless the content is a standalone header, title, or isolated fact.\n\n"
            "NOTE:\n"
            "You are NOT required to compute exact token counts.\n"
            "Final size enforcement will be handled by downstream processing.\n\n"
            "### STRUCTURAL CONSTRAINTS\n\n"
            "- Never place a breakpoint inside:\n"
            "  - Code blocks (``` ... ```)\n"
            "  - Markdown tables\n"
            "  Treat these as indivisible semantic units.\n\n"
            "### OUTPUT FORMAT\n\n"
            "Return strictly valid JSON only.\n\n"
            "{\n"
            "  \"breakpoints\": [5, 12, 28],\n"
            "  \"notes\": [\n"
            "    \"Sentence 5 begins a new authentication topic.\",\n"
            "    \"Sentence 12 introduces a new section header.\"\n"
            "  ]\n}"
        )

        self.__client: OpenAI = OpenAI(api_key=self._config.api_key)

    def determine_breakpoints(self, sentences: list[str]) -> Breakpoints:
        """Analyze sentences and determine where semantic chunk boundaries should be placed.
        
        Args:
            sentences: List of sentences to analyze for breakpoint placement.
            
        Returns:
            Breakpoints object containing indices and explanatory notes.
            
        Raises:
            RuntimeError: If OpenAI API call fails or response cannot be parsed.
        """
        try:
            response: Response = self._client.responses.parse(
                instructions=self._system_prompt,
                model=self._config.model_id,
                reasoning={"effort": "medium"},
                text_format=Breakpoints,
                input="\n".join([f"[{i}] {sentence}" for i, sentence in enumerate(sentences)]))
            breakpoints_data: Breakpoints | None = response.output_parsed
            if not breakpoints_data:
                raise ValueError("Failed to parse breakpoints from the model response.")
            return breakpoints_data
        except Exception as e:
            raise RuntimeError(f"Failed to determine breakpoints using OpenAI API: {e}") from e
        
        

#endregion


#region OllamaDocumentSegmenter class - handles document segmentation using Ollama LLM, to imitate the semantic chunking

class OllamaDocumentSegmenter(AICore[Breakpoints]):
    """Handles document segmentation using Ollama LLM for semantic chunking.
    
    Implements semantic chunk boundary detection using Ollama models.
    Extends AICore to provide structured breakpoint detection from chat responses.
    """

    def __init__(self) -> None:
        """Initialize Ollama document segmenter with configuration and system prompt."""
        config: OllamaAIConfig = OllamaAIConfig()
        system_prompt: str = (
            "You are a Document Segmentation Engine implementing semantic chunk boundary detection.\n\n"
            "Your task is to identify where new retrieval chunks should begin.\n"
            "You DO NOT output the chunks themselves - only the sentence indices where a new chunk starts.\n\n"
            "### CORE LOGIC\n\n"
            "1. Semantic Continuity\n"
            "- Read the document sentence by sentence.\n"
            "- Maintain a running notion of the current topic or line of reasoning.\n"
            "- As long as new sentences extend or elaborate the same idea, keep them in the same chunk.\n\n"
            "2. Topic Shift Detection\n"
            "Insert a breakpoint when there is a clear semantic transition, such as:\n"
            "- A new section or heading\n"
            "- A change in subject, goal, or reasoning\n"
            "- Discourse markers like 'Moving on', 'In contrast', 'Chapter 2'\n"
            "High cohesion markers ('Therefore', 'Additionally') indicate continuation.\n\n"
            "3. Size Awareness (Approximate)\n"
            "- Target chunk size: ~200-450 tokens (approximate).\n"
            "- If a chunk is clearly becoming too large, prefer inserting a breakpoint at the nearest logical pause (sentence boundary, paragraph break, or section boundary).\n"
            "- Avoid creating chunks smaller than ~50 tokens unless the content is a standalone header, title, or isolated fact.\n\n"
            "NOTE:\n"
            "You are NOT required to compute exact token counts.\n"
            "Final size enforcement will be handled by downstream processing.\n\n"
            "### STRUCTURAL CONSTRAINTS\n\n"
            "- Never place a breakpoint inside:\n"
            "  - Code blocks (``` ... ```)\n"
            "  - Markdown tables\n"
            "  Treat these as indivisible semantic units.\n\n"
            "### OUTPUT FORMAT\n\n"
            "Return strictly valid JSON only.\n\n"
            "{\n"
            "  \"breakpoints\": [5, 12, 28],\n"
            "  \"notes\": [\n"
            "    \"Sentence 5 begins a new authentication topic.\",\n"
            "    \"Sentence 12 introduces a new section header.\"\n"
            "  ]\n"
            "}"
        )
        super().__init__(system_behavior=system_prompt, config=config)

    def determine_breakpoints(self, sentences: list[str]) -> Breakpoints:
        """Analyze sentences and determine where semantic chunk boundaries should be placed.
        
        Args:
            sentences: List of sentences to analyze for breakpoint placement.
            
        Returns:
            Breakpoints object containing indices and explanatory notes.
            
        Raises:
            RuntimeError: If Ollama API call fails or response cannot be parsed.
        """
        try:
            input_text: str = "\n".join([f"[{i}] {sentence}" for i, sentence in enumerate(sentences)])
            return self.ask(input_text)
        except Exception as e:
            raise RuntimeError(f"Failed to determine breakpoints using Ollama: {e}") from e

    def _process_response(self, response: ChatResponse) -> Breakpoints:
        """Process the chat response and extract breakpoints.
        
        Args:
            response: Chat response from Ollama.
            
        Returns:
            Breakpoints object parsed from response.
            
        Raises:
            ValueError: If response cannot be parsed into Breakpoints.
        """
        try:
            content: str = response.message.content if response.message.content else ""
            
            # Extract JSON from response (handle markdown code blocks)
            json_str: str = content.strip()
            if json_str.startswith("```"):
                # Remove markdown code block markers
                lines = json_str.split("\n")
                json_str = "\n".join(lines[1:-1]) if len(lines) > 2 else json_str
            
            # Parse JSON and validate with Pydantic
            data = json.loads(json_str)
            return Breakpoints(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse breakpoints from Ollama response: {e}") from e


#endregion


#region PdfChunker class - placeholder for PDF chunking logic to execute the chunking process using OpenAI LLM

class PdfChunker:
    """Handles PDF document chunking using semantic segmentation.
    
    Extracts text from PDF files, splits into sentences, determines semantic breakpoints,
    and generates text chunks suitable for retrieval-augmented generation (RAG) systems.
    """

    @property
    def _segmenter(self) -> DocumentSegmenter:
        return self.__segmenter

    def __init__(self) -> None:
        self.__segmenter: DocumentSegmenter = DocumentSegmenter()

    def chunk_pdf(self, pdf_path: str) -> list[str]:
        """Process a PDF file and split it into semantic chunks.
        
        Args:
            pdf_path: Path to the PDF file to process.
            
        Returns:
            List of text chunks, each representing a semantically cohesive unit.
            
        Raises:
            FileNotFoundError: If PDF file does not exist.
            RuntimeError: If PDF extraction or chunking fails.
        """
        # Placeholder for PDF chunking logic using self._segmenter
        pdf_raw_text: str = self.__extract_text_from_pdf(pdf_path)
        sentences: list[str] = self.__split_text_into_sentences(pdf_raw_text)
        breakpoints = self.__determine_breakpoints(sentences)
        return self.__apply_breakpoints_to_sentences(sentences, breakpoints)

    def __extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            with open(pdf_path, "rb") as file:
                reader: PdfReader = PdfReader(file)
                text: str = ""
                for page in reader.pages:
                    text += f"{page.extract_text()}\n"
            return text
        except FileNotFoundError:
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from PDF: {e}") from e

    def __split_text_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences using NLTK's sentence tokenizer."""
        try:
            return nltk.sent_tokenize(text)
        except LookupError:
            # Download punkt tokenizer if not already present
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            return nltk.sent_tokenize(text)

    def __determine_breakpoints(self, sentences: list[str]) -> Breakpoints:
        return self._segmenter.determine_breakpoints(sentences)

    def __apply_breakpoints_to_sentences(self, sentences: list[str], breakpoints: Breakpoints) -> list[str]:
        # Validate and sanitize breakpoints
        valid_breakpoints = sorted(set(
            bp for bp in breakpoints.breakpoints 
            if 0 < bp < len(sentences)
        ))
        
        # If no valid breakpoints, return all text as one chunk
        if not valid_breakpoints:
            return [' '.join(sentences)]
        
        chunks: list[str] = []
        start_idx: int = 0
        
        for breakpoint in valid_breakpoints:
            # Skip if breakpoint equals start_idx (would create empty chunk)
            if breakpoint > start_idx:
                chunk: str = ' '.join(sentences[start_idx:breakpoint])
                chunks.append(chunk)
                start_idx = breakpoint
        
        # Add the remaining sentences as the last chunk
        if start_idx < len(sentences):
            chunk: str = ' '.join(sentences[start_idx:])
            chunks.append(chunk)
            
        return chunks

#endregion


#region OllamaPdfChunker class - handles PDF chunking using Ollama-based semantic segmentation

class OllamaPdfChunker:
    """Handles PDF document chunking using Ollama-based semantic segmentation.
    
    Extracts text from PDF files, splits into sentences, determines semantic breakpoints
    using Ollama, and generates text chunks suitable for retrieval-augmented generation (RAG) systems.
    """

    @property
    def _segmenter(self) -> OllamaDocumentSegmenter:
        """Get the Ollama document segmenter."""
        return self.__segmenter

    def __init__(self) -> None:
        """Initialize with Ollama document segmenter."""
        self.__segmenter: OllamaDocumentSegmenter = OllamaDocumentSegmenter()

    def chunk_pdf(self, pdf_path: str) -> list[str]:
        """Process a PDF file and split it into semantic chunks.
        
        Args:
            pdf_path: Path to the PDF file to process.
            
        Returns:
            List of text chunks, each representing a semantically cohesive unit.
            
        Raises:
            FileNotFoundError: If PDF file does not exist.
            RuntimeError: If PDF extraction or chunking fails.
        """
        pdf_raw_text: str = self.__extract_text_from_pdf(pdf_path)
        sentences: list[str] = self.__split_text_into_sentences(pdf_raw_text)
        breakpoints = self.__determine_breakpoints(sentences)
        return self.__apply_breakpoints_to_sentences(sentences, breakpoints)

    def __extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF file.
        
        Args:
            pdf_path: Path to PDF file.
            
        Returns:
            Extracted text content.
            
        Raises:
            FileNotFoundError: If PDF file not found.
            RuntimeError: If extraction fails.
        """
        try:
            with open(pdf_path, "rb") as file:
                reader: PdfReader = PdfReader(file)
                text: str = ""
                for page in reader.pages:
                    text += f"{page.extract_text()}\n"
            return text
        except FileNotFoundError:
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from PDF: {e}") from e

    def __split_text_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences using NLTK's sentence tokenizer.
        
        Args:
            text: Text to split.
            
        Returns:
            List of sentences.
        """
        try:
            return nltk.sent_tokenize(text)
        except LookupError:
            # Download punkt tokenizer if not already present
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            return nltk.sent_tokenize(text)

    def __determine_breakpoints(self, sentences: list[str]) -> Breakpoints:
        """Determine semantic breakpoints using Ollama segmenter.
        
        Args:
            sentences: List of sentences to analyze.
            
        Returns:
            Breakpoints object with indices and notes.
        """
        return self._segmenter.determine_breakpoints(sentences)

    def __apply_breakpoints_to_sentences(self, sentences: list[str], breakpoints: Breakpoints) -> list[str]:
        """Apply breakpoints to create text chunks.
        
        Args:
            sentences: List of sentences.
            breakpoints: Breakpoints object with chunk boundaries.
            
        Returns:
            List of text chunks.
        """
        # Validate and sanitize breakpoints
        valid_breakpoints = sorted(set(
            bp for bp in breakpoints.breakpoints 
            if 0 < bp < len(sentences)
        ))
        
        # If no valid breakpoints, return all text as one chunk
        if not valid_breakpoints:
            return [' '.join(sentences)]
        
        chunks: list[str] = []
        start_idx: int = 0
        
        for breakpoint in valid_breakpoints:
            # Skip if breakpoint equals start_idx (would create empty chunk)
            if breakpoint > start_idx:
                chunk: str = ' '.join(sentences[start_idx:breakpoint])
                chunks.append(chunk)
                start_idx = breakpoint
        
        # Add the remaining sentences as the last chunk
        if start_idx < len(sentences):
            chunk: str = ' '.join(sentences[start_idx:])
            chunks.append(chunk)
            
        return chunks


#endregion


#region TitanEmbeddingGenerator class - placeholder for embedding generation logic using AWS Titan model. Vectors will be stored in S3 bucket.

import boto3
import json

from typing import Any

from mypy_boto3_bedrock_runtime.client import BedrockRuntimeClient
from mypy_boto3_s3vectors.client import S3VectorsClient

class TitanEmbeddingGenerator:
    """Generates embeddings using AWS Titan model and stores them in S3 Vectors.
    
    Uses Amazon Bedrock's Titan Embed Text V2 model to generate vector embeddings
    for text chunks, then stores them in AWS S3 Vectors for similarity search.
    
    Attributes:
        __REGION_NAME: AWS region for Bedrock and S3Vectors services.
        __BUCKET_NAME: S3 Vectors bucket name for storing embeddings.
        __VECTOR_INDEX_NAME: Name of the vector index within the bucket.
        __EMBEDDING_MODEL_ID: Bedrock model ID for generating embeddings.
    """

    __REGION_NAME: str = ""
    __BUCKET_NAME: str = ""
    __VECTOR_INDEX_NAME: str = ""
    __EMBEDDING_MODEL_ID: str = ""


    @property
    def _bedrock_runtime(self) -> BedrockRuntimeClient:
        return self.__bedrock_runtime

    @property
    def _s3_vectors(self) -> S3VectorsClient:
        return self.__s3_vectors

    def __init__(self) -> None:
        self.__bedrock_runtime: BedrockRuntimeClient = boto3.client(service_name='bedrock-runtime', region_name=self.__REGION_NAME)
        self.__s3_vectors: S3VectorsClient = boto3.client(service_name='s3vectors', region_name=self.__REGION_NAME)

    def vectorize_and_store(self, text_chunks: list[str]) -> None:
        """Generate embeddings for text chunks and store them in S3 Vectors.
        
        Args:
            text_chunks: List of text chunks to vectorize and store.
            
        Raises:
            RuntimeError: If embedding generation or storage fails.
        """
        accumulated_embeddings: list[list[float]] = []
        for chunk in text_chunks:
            accumulated_embeddings.append(self.__generate_embedding(chunk))

        self.__store_embeddings_in_s3_vectors(text_chunks, accumulated_embeddings)

    def __generate_embedding(self, text: str) -> list[float]:
        try:
            response = self._bedrock_runtime.invoke_model(
                modelId=self.__EMBEDDING_MODEL_ID,
                contentType='application/json',
                accept='application/json',
                body=json.dumps({"inputText": text})
            )
            response_body = json.loads(response.get('body').read())
            embedding: list[float] | None = response_body.get('embedding')
            if embedding is None:
                raise ValueError("No embedding returned from the model")
            return embedding
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding for text: {e}") from e

    def __store_embeddings_in_s3_vectors(self, texts: list[str], embeddings: list[list[float]]) -> None:
        vectors_data: Any = [
            {
                "key": f"chunk-{i}",
                "data": {
                    "float32": embedding
                }, 
                "metadata": {
                    "chunk_id": str(i),
                    "text_preview": text[:500] + ("..." if len(text) > 500 else ""),
                    "char_length": str(len(text)),
                    "word_count": str(len(text.split()))
                }
            }
            for i, (text, embedding) in enumerate(zip(texts, embeddings))
        ]

        try:
            self._s3_vectors.put_vectors(
                vectorBucketName=self.__BUCKET_NAME,
                indexName=self.__VECTOR_INDEX_NAME,
                vectors=vectors_data)
        except Exception as e:
            raise RuntimeError(f"Failed to store vectors in S3Vectors bucket '{self.__BUCKET_NAME}': {e}") from e
        
#endregion