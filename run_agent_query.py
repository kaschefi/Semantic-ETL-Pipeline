# run_agent_query.py
import sys
from dotenv import load_dotenv

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

load_dotenv()

from src.agents.core import SemanticRAGAgent


def test_agent():
    agent = SemanticRAGAgent()
    user_query = "What is the difference between unit testing and integration testing in the test pyramid?"

    print(f"User Question: {user_query}\n")
    print("Agent executing Dual-Path Hybrid Search (Dense + BM25) and FlashRank Cross-Encoder reranking...")

    result = agent.answer_question_with_sources(user_query, namespace="benchmark-eval-sandbox")

    print("\n--- Grounded Answer ---")
    print(result["answer"])

    print("\n--- Source Citations ---")
    for s in result.get("sources", []):
        pages = ", ".join(s.get("source_pages", [])) or "N/A"
        print(f"[{s['citation_id']}] Category: {s['category']} | Pages: {pages} | Score: {s['score']:.4f}")
        print(f"    Summary: {s['summary']}")


if __name__ == "__main__":
    test_agent()
