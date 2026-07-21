import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
index_name = "semantic-etl-index"

pc = Pinecone(api_key=api_key)

if index_name not in pc.list_indexes().names():
    print(f"Creating a new Pinecone index: '{index_name}'...")

    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print("Index created successfully!")
else:
    print(f"Index '{index_name}' already exists and is ready to use.")