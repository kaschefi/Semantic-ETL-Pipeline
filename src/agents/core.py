# src/agent/core.py
from groq import Groq
from src.config import settings
from src.agents.retriever import ContextRetriever


class SemanticRAGAgent:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY) \
            if getattr(settings, 'GROQ_API_KEY', None) else None
        self.model_name = "openai/gpt-oss-120b"
        self.retriever = ContextRetriever()

    def answer_question(self, question: str, namespace: str = "documents", category_filter: str = None) -> str:
        """
        Answers a user question based strictly on injected semantic document insights.
        """
        filter_dict = {"category": category_filter} if category_filter else None

        hits = self.retriever.get_relevant_context(question, namespace=namespace, top_k=3, filter_dict=filter_dict)

        if not hits:
            return "I am sorry, but no matching context was located in the vector space database."

        context_blocks = []
        for i, hit in enumerate(hits):
            context_blocks.append(
                f"--- Context Block {i + 1} [Pages: {', '.join(hit['source_pages'])}] (Category: {hit['category']}) ---\n"
                f"Summary: {hit['summary']}\n"
                f"Content: {hit['text']}\n"
            )
        injected_context = "\n".join(context_blocks)

        system_prompt = (
            "You are an expert AI Knowledge Retrieval Agent. You must answer the user's question "
            "STRICTLY using the provided semantic context blocks below. If the answer cannot be found "
            "within the context blocks, state clearly that you do not possess the required data.\n\n"
            f"=== RETRIEVED KNOWLEDGE BACKGROUND ===\n{injected_context}\n======================================="
        )

        if not self.client:
            return f"Agent online, but Groq client is missing API keys.\n\n[Context Found]: {injected_context[:200]}..."

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.1
            )
            return completion.choices[0].message.content

        except Exception as e:
            return f" Agent failed to generate response via Groq: {str(e)}"