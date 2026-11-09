# src/agents/core.py
from typing import Optional, List, Dict, Any, AsyncIterator
from src.config import settings
from src.agents.retriever import ContextRetriever
from src.agents.reranker import ContextReranker
from src.providers.factory import get_llm_provider
from src.utils.logger import logger


class SemanticRAGAgent:
    """
    Grounded RAG Agent coordinating True Dual-Path Hybrid Retrieval, FlashRank Cross-Encoder
    reranking, inline citation attribution, and hallucination-resistant LLM synthesis.
    """
    def __init__(self):
        self.provider = get_llm_provider()
        self.retriever = ContextRetriever()
        self.reranker = ContextReranker()

    def _build_context_and_sources(self, hits: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
        """
        Formats retrieved and reranked hits into structured context blocks and serializable source records.
        """
        context_blocks = []
        sources = []

        for i, hit in enumerate(hits):
            ref_id = i + 1
            pages = ', '.join(hit.get('source_pages', [])) or 'N/A'
            category = hit.get('category', 'General')
            summary = hit.get('summary', '')
            text = hit.get('text', '')

            context_blocks.append(
                f"--- Context Block [{ref_id}] [Source Pages: {pages}] (Category: {category}) ---\n"
                f"Summary: {summary}\n"
                f"Content: {text}\n"
            )

            sources.append({
                "citation_id": ref_id,
                "category": category,
                "source_pages": hit.get('source_pages', []),
                "summary": summary,
                "score": hit.get('score', 0.0),
                "text_snippet": text[:200] + "..." if len(text) > 200 else text
            })

        injected_context = "\n".join(context_blocks)
        return injected_context, sources

    def _build_system_prompt(self, injected_context: str) -> str:
        return (
            "You are an expert AI Knowledge Retrieval Agent. You must answer the user's question "
            "STRICTLY using the provided semantic context blocks below.\n\n"
            "CITATION & GROUNDING RULES:\n"
            "1. Base all factual assertions directly on the context blocks.\n"
            "2. Cite your sources using bracketed references like [1], [2] immediately following the facts they support.\n"
            "3. If the answer cannot be fully determined from the provided context blocks, explicitly state: "
            "'The provided document context does not contain sufficient information to answer this question.'\n"
            "4. Do not invent, speculate, or extrapolate facts outside the context blocks.\n\n"
            f"=== RETRIEVED KNOWLEDGE BACKGROUND ===\n{injected_context}\n======================================="
        )

    def answer_question_with_sources(
        self,
        question: str,
        namespace: str = "documents",
        category_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Answers a user question based strictly on injected semantic document insights.
        Returns both the answer text and structured source citations.
        """
        filter_dict = {"category": category_filter} if category_filter else None

        # Step 1: Hybrid Retrieval fetching candidate pool
        candidate_hits = self.retriever.get_relevant_context(
            question,
            namespace=namespace,
            top_k=8,
            filter_dict=filter_dict
        )

        if not candidate_hits:
            return {
                "answer": "The provided document context does not contain sufficient information to answer this question.",
                "sources": []
            }

        # Step 2: FlashRank Cross-Encoder reranking candidates down to top 3 highest-quality context blocks
        hits = self.reranker.rerank(question, candidate_hits, top_n=3)
        injected_context, sources = self._build_context_and_sources(hits)
        system_prompt = self._build_system_prompt(injected_context)

        try:
            if hasattr(self.provider, "complete_text_sync"):
                answer = self.provider.complete_text_sync(question, system_prompt=system_prompt)
            else:
                import asyncio
                answer = asyncio.run(self.provider.complete_text_async(question, system_prompt=system_prompt))
            return {
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"Generation failure: {e}")
            return {
                "answer": f"Agent failed to generate response: {str(e)}",
                "sources": sources
            }

    def answer_question(
        self,
        question: str,
        namespace: str = "documents",
        category_filter: Optional[str] = None
    ) -> str:
        """
        Backward-compatible string-returning query method.
        """
        result = self.answer_question_with_sources(question, namespace, category_filter)
        return result["answer"]

    async def answer_question_stream(
        self,
        question: str,
        namespace: str = "documents",
        category_filter: Optional[str] = None
    ) -> AsyncIterator[str]:
        """
        Asynchronously streams grounded tokens with sub-800ms Time-to-First-Token (TTFT).
        """
        filter_dict = {"category": category_filter} if category_filter else None

        # 1. Dual-Path Retrieval
        candidate_hits = self.retriever.get_relevant_context(
            question,
            namespace=namespace,
            top_k=8,
            filter_dict=filter_dict
        )

        if not candidate_hits:
            yield "The provided document context does not contain sufficient information to answer this question."
            return

        # 2. FlashRank Cross-Encoder
        hits = self.reranker.rerank(question, candidate_hits, top_n=3)
        injected_context, _sources = self._build_context_and_sources(hits)
        system_prompt = self._build_system_prompt(injected_context)

        # 3. Stream from active LLM provider
        async for token in self.provider.stream_text_async(question, system_prompt=system_prompt):
            yield token
