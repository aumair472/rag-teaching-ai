"""
LLM service module.

Handles all interactions with OpenAI GPT-4o, including:
    - Synchronous generation
    - Streaming token yield (async generator)
    - System prompt construction for grounded RAG responses
"""

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ─── System Prompt ───────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an AI Teaching Assistant. Your role is to help students understand course material by providing clear, accurate answers.

STRICT RULES:
1. Answer ONLY from the provided context below. Do NOT use external knowledge.
2. If the context does not contain sufficient information to answer the question, respond with:
   "This topic is not covered in the course material."
3. Always cite your sources using the format: [Source: <source_name>, <page/slide/timestamp>]
4. Be concise but thorough. Use bullet points and structured formatting when appropriate.
5. If the student asks a follow-up question, use the conversation history to maintain continuity.
6. Never fabricate, hallucinate, or speculate beyond what is provided in the context.

CONTEXT:
{context}

CONVERSATION HISTORY:
{history}
"""


class LLMService:
    """
    Async LLM service for GPT-4o generation.

    Supports both standard and streaming response modes.

    Attributes:
        client: Async OpenAI client.
        model: Model name (e.g., ``gpt-4o``).
        temperature: Sampling temperature.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> None:
        settings = get_settings()
        self.model = model or settings.openai_model
        self.temperature = temperature if temperature is not None else settings.openai_temperature

        self.client = AsyncOpenAI(api_key=api_key or settings.openai_api_key)

        logger.info(
            "LLM service initialized",
            extra={"model": self.model, "temperature": self.temperature},
        )

    def _build_messages(
        self,
        question: str,
        context: str,
        history: str = "",
    ) -> List[Dict[str, str]]:
        """
        Build the message array for the chat completion.

        Args:
            question: The student's question.
            context: Concatenated context from retrieval.
            history: Formatted conversation history.

        Returns:
            A list of message dicts for the OpenAI API.
        """
        system_content = SYSTEM_PROMPT.format(
            context=context or "No relevant context found.",
            history=history or "No previous conversation.",
        )
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": question},
        ]

    async def generate(
        self,
        question: str,
        context: str,
        history: str = "",
    ) -> str:
        """
        Generate a complete response (non-streaming).

        Args:
            question: The student's question.
            context: Retrieved context passages.
            history: Conversation history string.

        Returns:
            The generated answer string.

        Raises:
            Exception: If the OpenAI API call fails.
        """
        messages = self._build_messages(question, context, history)

        logger.info("Generating LLM response", extra={"model": self.model})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=1500,
        )

        answer = response.choices[0].message.content or ""

        logger.info(
            "LLM response generated",
            extra={
                "tokens_prompt": response.usage.prompt_tokens if response.usage else 0,
                "tokens_completion": response.usage.completion_tokens if response.usage else 0,
            },
        )

        return answer.strip()

    async def generate_stream(
        self,
        question: str,
        context: str,
        history: str = "",
    ) -> AsyncGenerator[str, None]:
        """
        Stream response tokens as an async generator.

        Yields individual content deltas as they arrive from the API.

        Args:
            question: The student's question.
            context: Retrieved context passages.
            history: Conversation history string.

        Yields:
            String tokens from the LLM response.
        """
        messages = self._build_messages(question, context, history)

        logger.info("Streaming LLM response", extra={"model": self.model})

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=1500,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    def format_context(
        self,
        chunks: List[Dict[str, Any]],
    ) -> str:
        """
        Format retrieved chunks into a context string for the prompt.

        Args:
            chunks: List of chunk dictionaries with ``text`` and metadata.

        Returns:
            A formatted context string with source citations.
        """
        if not chunks:
            return "No relevant context found."

        parts: List[str] = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source_name", "Unknown")
            source_type = chunk.get("source_type", "unknown")

            # Build location string
            location_parts: List[str] = []
            if chunk.get("page"):
                location_parts.append(f"Page {chunk['page']}")
            if chunk.get("slide"):
                location_parts.append(f"Slide {chunk['slide']}")
            if chunk.get("timestamp"):
                location_parts.append(f"Timestamp {chunk['timestamp']}")

            location = ", ".join(location_parts) if location_parts else "N/A"
            text = chunk.get("text", "")

            parts.append(
                f"[{i}] Source: {source} ({source_type}) | {location}\n{text}"
            )

        return "\n\n---\n\n".join(parts)
