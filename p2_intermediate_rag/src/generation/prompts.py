"""
Prompt Builder
==============
Build prompts for RAG generation.
"""

from typing import Any, Dict, List, Optional


class PromptBuilder:
    """
    Build prompts for RAG generation with customizable templates.
    """
    
    DEFAULT_SYSTEM_TEMPLATE = """You are a helpful AI assistant. Answer questions based ONLY on the provided context.
If the answer cannot be found in the context, say "I don't have enough information to answer this question."
Always cite your sources by mentioning which document the information came from."""
    
    DEFAULT_QA_TEMPLATE = """Context:
{context}

Question: {question}

Instructions:
1. Answer based ONLY on the context above
2. If unsure, say so
3. Cite the source documents

Answer:"""
    
    def __init__(
        self,
        system_template: Optional[str] = None,
        qa_template: Optional[str] = None,
    ):
        """
        Initialize the prompt builder.
        
        Args:
            system_template: System message template
            qa_template: QA template with {context} and {question} placeholders
        """
        self.system_template = system_template or self.DEFAULT_SYSTEM_TEMPLATE
        self.qa_template = qa_template or self.DEFAULT_QA_TEMPLATE
    
    def build_context(
        self,
        results: List[Dict[str, Any]],
        include_source: bool = True,
        include_score: bool = False,
    ) -> str:
        """
        Build context string from retrieval results.
        
        Args:
            results: List of retrieval results
            include_source: Whether to include source in context
            include_score: Whether to include relevance score
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, result in enumerate(results, 1):
            text = result.get("text", "")
            metadata = result.get("metadata", {})
            
            # Build header
            header_parts = [f"[{i}]"]
            
            if include_source:
                source = metadata.get("source", metadata.get("filename", "Unknown"))
                header_parts.append(f"Source: {source}")
                
                if "chunk_index" in metadata:
                    header_parts.append(f"Chunk: {metadata['chunk_index']}")
            
            if include_score:
                score = result.get("rerank_score", result.get("combined_score", result.get("distance", 0)))
                header_parts.append(f"Score: {score:.3f}")
            
            header = " | ".join(header_parts)
            context_parts.append(f"{header}\n{text}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def build_prompt(
        self,
        question: str,
        results: List[Dict[str, Any]],
        include_system: bool = True,
    ) -> str:
        """
        Build the full prompt for generation.
        
        Args:
            question: User question
            results: Retrieval results
            include_system: Whether to include system message
            
        Returns:
            Complete prompt string
        """
        # Build context from results
        context = self.build_context(results)
        
        # Build QA prompt (use replace to avoid KeyError when user input contains { or })
        qa_prompt = self.qa_template.replace("{context}", context).replace("{question}", question)
        
        # Combine with system message if requested
        if include_system:
            return f"{self.system_template}\n\n{qa_prompt}"
        
        return qa_prompt
    
    def build_chat_messages(
        self,
        question: str,
        results: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """
        Build messages for chat-based APIs.
        
        Args:
            question: Current question
            results: Retrieval results
            chat_history: Previous messages (optional)
            
        Returns:
            List of message dicts with 'role' and 'content'
        """
        messages = []
        
        # System message
        messages.append({
            "role": "system",
            "content": self.system_template,
        })
        
        # Add chat history if provided
        if chat_history:
            messages.extend(chat_history)
        
        # Build user message with context (use replace to avoid KeyError when user input contains { or })
        context = self.build_context(results)
        user_content = self.qa_template.replace("{context}", context).replace("{question}", question)
        
        messages.append({
            "role": "user",
            "content": user_content,
        })
        
        return messages
