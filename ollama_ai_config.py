"""Configuration for Ollama AI."""

from dotenv import load_dotenv, find_dotenv
from os import getenv


class OllamaAIConfig:
    """Configuration manager for Ollama AI model settings.
    
    Loads model ID from environment variables using python-dotenv.
    Optionally loads base URL if OLLAMA_BASE_URL is set in .env file.
    
    Raises:
        ValueError: If required environment variables are not set.
    """

    @property
    def model_id(self) -> str:
        """Get the Ollama model ID."""
        return self.__model_id

    @property
    def base_url(self) -> str | None:
        """Get the Ollama base URL (optional)."""
        return self.__base_url

    def __init__(self) -> None:
        """Initialize configuration from environment variables."""
        load_dotenv(find_dotenv(), override=True)
        self.__model_id: str = self.__get_model_id()
        self.__base_url: str | None = getenv("OLLAMA_BASE_URL")

    def __get_model_id(self) -> str:
        """Get model ID from environment variable.
        
        Returns:
            Model ID string.
            
        Raises:
            ValueError: If OLLAMA_MODEL_ID is not set.
        """
        value: str | None = getenv("OLLAMA_MODEL_ID")
        if not value:
            raise ValueError("OLLAMA_MODEL_ID environment variable is not set.")
        return value
