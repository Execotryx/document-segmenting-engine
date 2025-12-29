"""Core AI functionality using Ollama."""

from ollama import chat, ChatResponse, Message
from ollama_ai_config import OllamaAIConfig
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

TAiResponse = TypeVar("TAiResponse", default=Any)


class HistoryManager:
    """Manage chat history and system behavior.
    
    Maintains conversation history and system instructions for Ollama chat sessions.
    Ensures system message is always first in chat history.
    """
    
    @property
    def system_behavior(self) -> str:
        """System instruction for the conversation."""
        return self.__system_behavior

    @property
    def chat_history(self) -> list[Message]:
        """Chat history for the conversation.
        
        Returns:
            List of messages with system message always first.
        """
        if not self.__chat_history or self.__chat_history[0].get("role") != "system":
            self.__chat_history.insert(0, self.__create_message_with_role("system", self.system_behavior))
        return self.__chat_history

    @property
    def config(self) -> OllamaAIConfig:
        """Get the Ollama AI configuration."""
        return self.__config

    def __init__(self, system_behavior: str, config: OllamaAIConfig) -> None:
        """Create a HistoryManager with the given system behavior.

        Parameters:
            system_behavior: System instruction string.
            config: Ollama AI configuration.
        """
        self.__system_behavior: str = system_behavior
        self.__chat_history: list[Message] = []
        self.__config: OllamaAIConfig = config

    def add_user_message(self, message: str) -> None:
        """Add a user message to the chat history.
        
        Args:
            message: User message content.
        """
        self.__chat_history.append(self.__create_message_with_role("user", message))

    def add_assistant_message(self, message: str) -> None:
        """Add an assistant message to the chat history.
        
        Args:
            message: Assistant message content.
        """
        self.__chat_history.append(self.__create_message_with_role("assistant", message))

    def __create_message_with_role(self, role: str, content: str) -> Message:
        """Create a message with the given role and content.
        
        Args:
            role: Message role (system, user, or assistant).
            content: Message content.
            
        Returns:
            Message dictionary with role and content.
        """
        return Message(role=role, content=content)


class AICore(ABC, Generic[TAiResponse]):
    """Abstract base class for AI core functionality using Ollama.
    
    Provides common chat interface with history management.
    Subclasses must implement _process_response to handle model responses.
    """

    @property
    def _config(self) -> OllamaAIConfig:
        """Get the Ollama AI configuration."""
        return self.__config

    @property
    def _history_manager(self) -> HistoryManager:
        """Get the history manager."""
        return self.__history_manager

    def __init__(self, system_behavior: str, config: OllamaAIConfig) -> None:
        """Initialize AI core with system behavior and configuration.
        
        Args:
            system_behavior: System instruction for the AI.
            config: Ollama AI configuration.
        """
        self.__config: OllamaAIConfig = config
        self.__history_manager: HistoryManager = HistoryManager(system_behavior, self._config)

    def ask(self, request: str) -> TAiResponse:
        """Send a request to the AI and get a response.
        
        Args:
            request: User request/question.
            
        Returns:
            Processed response of type TAiResponse.
            
        Raises:
            RuntimeError: If chat request fails.
        """
        self._history_manager.add_user_message(request)
        
        try:
            response: ChatResponse = chat(
                model=self._config.model_id,
                messages=self._history_manager.chat_history
            )
            
            response_content = response.message.content if response.message.content else "No response was received."
            self._history_manager.add_assistant_message(response_content)
            
            return self._process_response(response)
        except Exception as e:
            raise RuntimeError(f"Failed to get response from Ollama: {e}") from e

    @abstractmethod
    def _process_response(self, response: ChatResponse) -> TAiResponse:
        """Process the chat response and return typed result.
        
        Args:
            response: Chat response from Ollama.
            
        Returns:
            Processed response of type TAiResponse.
        """
        pass
