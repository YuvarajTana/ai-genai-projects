"""
Ollama LLM Backend
==================
Generate responses using local Ollama server.
"""

import json
from typing import Iterator, Optional

import requests
from loguru import logger

from .base import BaseLLM, GenerationConfig


class OllamaLLM(BaseLLM):
    """
    LLM backend using Ollama for local model inference.
    """
    
    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: int = 300,
    ):
        """
        Initialize Ollama LLM.
        
        Args:
            model: Model name (e.g., llama3.2, mistral, etc.)
            base_url: Ollama server URL
            timeout: Request timeout in seconds
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.generate_url = f"{self.base_url}/api/generate"
    
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        Generate a response.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            
        Returns:
            Generated text
        """
        config = config or GenerationConfig()
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "num_predict": config.max_tokens,
                "top_p": config.top_p,
            },
        }
        
        if config.stop_sequences:
            payload["options"]["stop"] = config.stop_sequences
        
        logger.debug("Generating with model {} (prompt length: {})", self.model, len(prompt))
        try:
            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()
            response_text = result.get("response", "")
            logger.debug("Generated {} chars", len(response_text))
            return response_text
        
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: `ollama serve`"
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Ollama request timed out after {self.timeout}s"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")
    
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
        config = config or GenerationConfig()
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": config.temperature,
                "num_predict": config.max_tokens,
                "top_p": config.top_p,
            },
        }
        
        if config.stop_sequences:
            payload["options"]["stop"] = config.stop_sequences
        
        try:
            response = requests.post(
                self.generate_url,
                json=payload,
                stream=True,
                timeout=self.timeout,
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                    if data.get("done", False):
                        break
        
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: `ollama serve`"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama streaming failed: {e}")
    
    def list_models(self) -> list:
        """
        List available models on the Ollama server.
        
        Returns:
            List of model names
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10,
            )
            response.raise_for_status()
            models = response.json().get("models", [])
            return [m["name"] for m in models]
        except Exception:
            return []
    
    def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            return response.status_code == 200
        except Exception:
            return False
