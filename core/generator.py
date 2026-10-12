"""
Generation Engine for RAG Pipeline.
Synthesizes answers based on retrieved context chunks and injects precise source citations.
Supports external LLMs (OpenAI, Groq) with an offline extractive/synthesizing fallback.
"""

import logging
import os
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class RAGAnswerGenerator:
    """
    Generates answers from retrieved document chunks with citation tracking.
    """

    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.groq_key = os.getenv("GROQ_API_KEY")

    def _generate_with_openai(self, query: str, context_chunks: List[Dict]) -> Optional[str]:
        """Queries OpenAI API if configured."""
        if not self.openai_key:
            return None
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_key)

            context_str = "\n\n".join([
                f"Source {idx+1} {c['citation']}:\n{c['text']}"
                for idx, c in enumerate(context_chunks)
            ])

            system_prompt = (
                "You are an expert AI document research assistant. "
                "Answer the user question accurately using ONLY the provided context sources. "
                "For every fact or claim, include inline citations referencing the source, e.g. [Filename | Page X]. "
                "If the context does not contain the answer, explicitly state that the information is not present in the documents."
            )

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"},
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning("OpenAI generation failed: %s. Using local synthesis.", e)
            return None

    def _generate_with_groq(self, query: str, context_chunks: List[Dict]) -> Optional[str]:
        """Queries Groq API if configured."""
        if not self.groq_key:
            return None
        try:
            from groq import Groq
            client = Groq(api_key=self.groq_key)

            context_str = "\n\n".join([
                f"Source {idx+1} {c['citation']}:\n{c['text']}"
                for idx, c in enumerate(context_chunks)
            ])

            prompt = (
                "Answer the question accurately based on the provided context. "
                "Cite sources using [Filename | Page X].\n\n"
                f"Context:\n{context_str}\n\nQuestion: {query}"
            )

            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.warning("Groq generation failed: %s. Using local synthesis.", e)
            return None

    def _local_synthesize(self, query: str, context_chunks: List[Dict]) -> str:
        """
        High-precision contextual synthesis engine for zero-cost offline generation.
        Extracts key sentences answering the query and builds cited prose.
        """
        if not context_chunks:
            return "I could not find relevant information in the uploaded documents to answer your question."

        query_terms = set(re.findall(r"\w+", query.lower()))
        stop_words = {"what", "is", "the", "a", "an", "and", "in", "of", "to", "how", "why", "does", "do", "for", "are"}
        core_terms = query_terms - stop_words

        candidate_sentences = []
        for chunk in context_chunks:
            citation = chunk["citation"]
            # Split into sentences
            raw_sentences = re.split(r"(?<=[.!?])\s+", chunk["text"])
            for sentence in raw_sentences:
                clean_s = sentence.strip()
                if len(clean_s) < 20:
                    continue
                s_words = set(re.findall(r"\w+", clean_s.lower()))
                overlap = len(core_terms.intersection(s_words))
                if overlap > 0:
                    candidate_sentences.append({
                        "sentence": clean_s,
                        "citation": citation,
                        "overlap": overlap,
                        "chunk_score": chunk["cosine_similarity"],
                    })

        # Sort by relevance
        candidate_sentences.sort(key=lambda x: (x["overlap"], x["chunk_score"]), reverse=True)

        if not candidate_sentences:
            top_chunk = context_chunks[0]
            excerpt = " ".join(top_chunk["text"].split()[:60]) + "..."
            return (
                f"Based on the retrieved context from {top_chunk['citation']}, "
                f"the relevant material states:\n\n\"{excerpt}\""
            )

        # Pick top relevant sentences (up to 3 unique)
        chosen = []
        seen_texts = set()
        for item in candidate_sentences:
            if item["sentence"] not in seen_texts:
                seen_texts.add(item["sentence"])
                chosen.append(item)
                if len(chosen) >= 3:
                    break

        # Assemble cited answer
        prose_parts = []
        for item in chosen:
            prose_parts.append(f"{item['sentence']} {item['citation']}")

        synthesis = " ".join(prose_parts)
        return (
            f"Based on the provided documentation, {synthesis}"
        )

    def generate_answer(self, query: str, context_chunks: List[Dict]) -> Dict:
        """
        Main answering pipeline combining retrieval context, synthesis, and structured citations.
        """
        if not context_chunks:
            return {
                "query": query,
                "answer": "No relevant documents found. Please upload documents to ask questions.",
                "model": "None",
                "citations": [],
                "context_chunks_used": 0,
            }

        model_name = "Contextual RAG Synthesizer (Local)"
        answer_text = None

        # 1. Try external LLMs if keys are available
        if self.openai_key:
            answer_text = self._generate_with_openai(query, context_chunks)
            if answer_text:
                model_name = "OpenAI gpt-4o-mini"

        if not answer_text and self.groq_key:
            answer_text = self._generate_with_groq(query, context_chunks)
            if answer_text:
                model_name = "Groq LLaMA-3.1"

        # 2. Fallback to local synthesizer
        if not answer_text:
            answer_text = self._local_synthesize(query, context_chunks)

        # 3. Format structured citations
        citations = []
        for c in context_chunks:
            citations.append({
                "source_filename": c["source_filename"],
                "page_number": c["page_number"],
                "chunk_index": c["chunk_index"],
                "cosine_similarity": c["cosine_similarity"],
                "citation_label": c["citation"],
                "excerpt": c["text"][:220] + "..." if len(c["text"]) > 220 else c["text"],
            })

        return {
            "query": query,
            "answer": answer_text,
            "model": model_name,
            "citations": citations,
            "context_chunks_used": len(context_chunks),
            "top_similarity": context_chunks[0]["cosine_similarity"] if context_chunks else 0.0,
        }


_GENERATOR_INSTANCE = None


def get_generator() -> RAGAnswerGenerator:
    """Returns the shared generator singleton."""
    global _GENERATOR_INSTANCE
    if _GENERATOR_INSTANCE is None:
        _GENERATOR_INSTANCE = RAGAnswerGenerator()
    return _GENERATOR_INSTANCE
