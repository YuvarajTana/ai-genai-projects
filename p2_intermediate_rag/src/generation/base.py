"""
Base LLM Interface
==================
Abstract base class for LLM backends.
"""

from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.1
    max_tokens: int = 2048
    top_p: float = 0.9
    stop_sequences: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stop": self.stop_sequences if self.stop_sequences else None,
        }


class BaseLLM(ABC):
    """
    Abstract base class for LLM backends.
    """
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        """
        Generate a streaming response.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            
        Yields:
            Generated text chunks
        """
        pass
    
    async def agenerate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        Async generate a response.
        
        Default implementation calls sync method.
        Override for true async support.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            
        Returns:
            Generated text
        """
        return await asyncio.to_thread(self.generate, prompt, config)
    
    async def agenerate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[str]:
        """
        Async streaming generation.
        
        Default implementation wraps sync method.
        Override for true async support.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            
        Yields:
            Generated text chunks
        """
        for chunk in self.generate_stream(prompt, config):
            yield chunk
