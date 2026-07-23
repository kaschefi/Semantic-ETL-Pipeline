# run_agent_query.py
from dotenv import load_dotenv

load_dotenv()

from src.agents.core import SemanticRAGAgent


def test_agent():
    agent = SemanticRAGAgent()

    # 💡 Ask a question matching your ingested mock document details!
    user_query = "what is a shell "

    print(f" User Question: {user_query}\n")
    print(" Agent is processing vector search and checking with Groq...")

    response = agent.answer_question(user_query, namespace="integration-testing-sandbox")

    print("\n Final Grounded Agent Response:")
    print(response)


if __name__ == "__main__":
    test_agent()