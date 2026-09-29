"""NLP tools using NLTK."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from aof.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_NLTK_READY = False


def _ensure_nltk() -> None:
    """Download required NLTK data on first use."""
    global _NLTK_READY
    if _NLTK_READY:
        return
    import nltk

    for resource in ["punkt_tab", "averaged_perceptron_tagger_eng", "maxent_ne_chunker_tab", "words"]:
        try:
            nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else resource)
        except LookupError:
            nltk.download(resource, quiet=True)
    _NLTK_READY = True


def register(registry: ToolRegistry) -> None:
    """Register NLP tools."""

    @registry.register_function(
        name="text_tokenize",
        description="Tokenize text into sentences and words. Returns sentence count, word count, and the tokenized sentences.",
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to tokenize",
                },
            },
            "required": ["text"],
        },
    )
    async def text_tokenize(text: str) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_tokenize, text)

    @registry.register_function(
        name="text_summarize",
        description="Extract the most important sentences from text using frequency-based ranking.",
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to summarize",
                },
                "num_sentences": {
                    "type": "integer",
                    "description": "Number of sentences to extract (default 3)",
                },
            },
            "required": ["text"],
        },
    )
    async def text_summarize(text: str, num_sentences: int = 3) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_summarize, text, num_sentences)

    @registry.register_function(
        name="text_entities",
        description="Extract named entities (people, organizations, locations) from text using NLTK NER.",
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to analyze",
                },
            },
            "required": ["text"],
        },
    )
    async def text_entities(text: str) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_entities, text)


def _sync_tokenize(text: str) -> dict:
    try:
        _ensure_nltk()
        from nltk.tokenize import sent_tokenize, word_tokenize

        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        return {
            "sentence_count": len(sentences),
            "word_count": len(words),
            "sentences": sentences[:20],  # cap output
        }
    except Exception as e:
        return {"error": f"Tokenization failed: {e}"}


def _sync_summarize(text: str, num_sentences: int) -> dict:
    """Extractive summarization using word frequency scoring."""
    try:
        _ensure_nltk()
        from nltk.tokenize import sent_tokenize, word_tokenize
        from nltk.probability import FreqDist

        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return {"summary": text, "sentence_count": len(sentences)}

        # Score sentences by word frequency
        words = word_tokenize(text.lower())
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
                       "to", "for", "of", "and", "or", "but", "it", "this", "that",
                       "with", "from", "by", "as", "be", "has", "have", "had", "not"}
        filtered = [w for w in words if w.isalnum() and w not in stop_words]
        freq = FreqDist(filtered)

        scored = []
        for i, sent in enumerate(sentences):
            sent_words = word_tokenize(sent.lower())
            score = sum(freq.get(w, 0) for w in sent_words if w.isalnum())
            scored.append((score, i, sent))

        scored.sort(reverse=True)
        top = sorted(scored[:num_sentences], key=lambda x: x[1])
        summary = " ".join(s[2] for s in top)

        return {"summary": summary, "sentence_count": len(sentences)}
    except Exception as e:
        return {"error": f"Summarization failed: {e}"}


def _sync_entities(text: str) -> dict:
    try:
        _ensure_nltk()
        import nltk
        from nltk.tokenize import word_tokenize

        tokens = word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        chunks = nltk.ne_chunk(tagged)

        entities: dict[str, list[str]] = {}
        for chunk in chunks:
            if hasattr(chunk, "label"):
                label = chunk.label()
                entity = " ".join(c[0] for c in chunk)
                entities.setdefault(label, [])
                if entity not in entities[label]:
                    entities[label].append(entity)

        return {"entities": entities, "count": sum(len(v) for v in entities.values())}
    except Exception as e:
        return {"error": f"NER failed: {e}"}
