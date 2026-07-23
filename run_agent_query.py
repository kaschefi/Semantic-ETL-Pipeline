# run_agent_query.py
import sys
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

from src.agents.core import SemanticRAGAgent


def test_agent():
    agent = SemanticRAGAgent()

    # 💡 Ask a question matching your ingested mock document details!
    user_query = "What are the system architecture requirements?"

    print(f" User Question: {user_query}\n")
    print(" Agent is processing vector search, hybrid scoring, and cross-encoder reranking...")

    response = agent.answer_question(user_query, namespace="integration-testing-sandbox")

    print("\n Final Grounded Agent Response:")
    print(response)


if __name__ == "__main__":
    test_agent()