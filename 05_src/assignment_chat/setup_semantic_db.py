#!/usr/bin/env python3
"""
setup_semantic_db.py

Generates embeddings for quotes and stores them in ChromaDB.
"""

import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv('../.secrets')

# Initialize OpenAI client
client = OpenAI(
    base_url='https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1',
    api_key='any value',
    default_headers={"x-api-key": os.getenv('API_GATEWAY_KEY')}
)

# Sample quotes dataset
quotes = [
    # Food (10)
    "A warm meal turns a hard day into something you can survive.",
    "The best seasoning is the feeling that someone made this for you.",
    "Good cooking is attention made edible.",
    "A shared table is a small, daily celebration.",
    "Comfort has a flavor, and it usually tastes like home.",
    "Food is memory you can hold in your hands.",
    "The first bite can be a doorway back to your happiest place.",
    "When the soup is right, the whole world softens a little.",
    "The simplest dish becomes extraordinary when it's made with care.",
    "A good meal doesn't just feed you—it reassures you.",

    # Friendship (10)
    "Real friends don't rescue you from storms; they sit beside you in the rain.",
    "Friendship is the quiet agreement to keep showing up.",
    "A friend is someone who makes your good news bigger and your bad news smaller.",
    "You can measure trust by how safe silence feels.",
    "The truest kindness is being understood without needing to perform.",
    "Friendship is laughing at the same nonsense for years and never getting tired of it.",
    "The best friendships feel like exhaling.",
    "A friend is a mirror that reflects your worth on days you forget it.",
    "Some people become home, even if you never share an address.",
    "Good friends help you carry life without making you feel heavy.",

    # Community (10)
    "Community is strangers becoming neighbors one small favor at a time.",
    "Belonging isn't found—it's built.",
    "A community is a chorus: different voices, one song.",
    "When one person is lifted, the whole street rises a little.",
    "Care is the infrastructure that holds a place together.",
    "A shared future starts with shared responsibility.",
    "The strongest neighborhoods are stitched together by everyday generosity.",
    "Community is the art of making room for one more.",
    "We are safer when we are seen.",
    "Together is a direction, not just a feeling.",
]

def get_embedding(text, model="text-embedding-3-small"):
    """Get embedding for a single text."""
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def setup_chroma_db():
    """Create ChromaDB collection and add quotes with embeddings."""
    # Initialize ChromaDB with persistence
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    # Create or get collection
    collection_name = "inspirational_quotes"
    try:
        collection = chroma_client.get_collection(name=collection_name)
        print("Collection already exists, skipping setup.")
        return
    except Exception:
        collection = chroma_client.create_collection(name=collection_name)

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = []
    for i, quote in enumerate(quotes):
        print(f"Processing quote {i+1}/{len(quotes)}")
        embedding = get_embedding(quote)
        embeddings.append(embedding)

    # Add to collection
    ids = [f"quote_{i}" for i in range(len(quotes))]
    collection.add(
        embeddings=embeddings,
        documents=quotes,
        ids=ids
    )

    print(f"Added {len(quotes)} quotes to ChromaDB collection '{collection_name}'")

if __name__ == "__main__":
    setup_chroma_db()