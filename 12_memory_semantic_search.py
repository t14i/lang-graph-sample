"""
LangGraph Memory Verification - Part 2: Semantic Search

Goal: Verify semantic search with embeddings
- InMemoryStore with index configuration
- OpenAI embeddings (text-embedding-3-small)
- Similarity search with scores

Reference: https://langchain-ai.github.io/langgraph/concepts/memory/
"""

import os
from langgraph.store.memory import InMemoryStore


def main():
    print("=" * 60)
    print("LangGraph Memory - Semantic Search")
    print("=" * 60)

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        print("Run with: uv run --env-file .env python 12_memory_semantic_search.py")
        return

    # =========================================================================
    # Test 1: Initialize store with embedding index
    # =========================================================================
    print("\n" + "-" * 40)
    print("Test 1: Initialize store with embedding index")
    print("-" * 40)

    store = InMemoryStore(
        index={
            "dims": 1536,  # text-embedding-3-small dimension
            "embed": "openai:text-embedding-3-small",
        }
    )
    print("InMemoryStore initialized with:")
    print("  - dims: 1536")
    print("  - embed: openai:text-embedding-3-small")

    # =========================================================================
    # Test 2: Store memories with different topics
    # =========================================================================
    print("\n" + "-" * 40)
    print("Test 2: Store memories with different topics")
    print("-" * 40)

    memories = [
        ("food_1", "I love Italian food, especially pasta and pizza"),
        ("food_2", "My favorite cuisine is Japanese, especially sushi and ramen"),
        ("food_3", "I'm allergic to shellfish and peanuts"),
        ("work_1", "I work as a software engineer at a tech company"),
        ("work_2", "I prefer working remotely from home"),
        ("hobby_1", "I enjoy playing guitar and piano in my free time"),
        ("hobby_2", "I like hiking and camping in the mountains"),
        ("schedule_1", "I usually wake up at 7 AM and start work at 9 AM"),
        ("schedule_2", "I prefer afternoon meetings over morning ones"),
    ]

    for key, text in memories:
        # Store with 'text' field for semantic indexing
        store.put(("memories", "user_123"), key, {"text": text, "category": key.split("_")[0]})
        print(f"  Stored: {key}")

    # =========================================================================
    # Test 3: Semantic search - food preferences
    # =========================================================================
    print("\n" + "-" * 40)
    print("Test 3: Semantic search - 'What food do I like?'")
    print("-" * 40)

    results = list(store.search(
        ("memories", "user_123"),
        query="What food do I like?",
        limit=3
    ))

    for i, item in enumerate(results, 1):
        print(f"  {i}. [{item.score:.4f}] {item.value['text']}")

    # =========================================================================
    # Test 4: Semantic search - work related
    # =========================================================================
    print("\n" + "-" * 40)
    print("Test 4: Semantic search - 'Tell me about my job'")
    print("-" * 40)

    results = list(store.search(
        ("memories", "user_123"),
        query="Tell me about my job",
        limit=3
    ))

    for i, item in enumerate(results, 1):
        print(f"  {i}. [{item.score:.4f}] {item.value['text']}")

    # =========================================================================
    # Test 5: Semantic search - schedule
    # =========================================================================
    print("\n" + "-" * 40)
    print("Test 5: Semantic search - 'When should we schedule a meeting?'")
    print("-" * 40)

    results = list(store.search(
        ("memories", "user_123"),
        query="When should we schedule a meeting?",
        limit=3
    ))

    for i, item in enumerate(results, 1):
        print(f"  {i}. [{item.score:.4f}] {item.value['text']}")

    # =========================================================================
    # Test 6: Semantic search - allergies (safety critical)
    # =========================================================================
    print("\n" + "-" * 40)
    print("Test 6: Semantic search - 'dietary restrictions'")
    print("-" * 40)

    results = list(store.search(
        ("memories", "user_123"),
        query="dietary restrictions or food allergies",
        limit=3
    ))

    for i, item in enumerate(results, 1):
        print(f"  {i}. [{item.score:.4f}] {item.value['text']}")

    # =========================================================================
    # Test 7: Filter by metadata
    # =========================================================================
    print("\n" + "-" * 40)
    print("Test 7: Search with filter (category=food)")
    print("-" * 40)

    results = list(store.search(
        ("memories", "user_123"),
        query="preferences",
        limit=5,
        filter={"category": "food"}
    ))

    print(f"  Found {len(results)} items with category='food'")
    for i, item in enumerate(results, 1):
        print(f"  {i}. [{item.score:.4f}] {item.value['text']}")

    # =========================================================================
    # Test 8: Similarity threshold analysis
    # =========================================================================
    print("\n" + "-" * 40)
    print("Test 8: Similarity score analysis")
    print("-" * 40)

    # High relevance query
    results_high = list(store.search(
        ("memories", "user_123"),
        query="pasta pizza italian",
        limit=5
    ))
    print("Query: 'pasta pizza italian'")
    for item in results_high:
        print(f"  [{item.score:.4f}] {item.value['text'][:50]}...")

    print()

    # Low relevance query
    results_low = list(store.search(
        ("memories", "user_123"),
        query="quantum physics",
        limit=5
    ))
    print("Query: 'quantum physics' (unrelated)")
    for item in results_low:
        print(f"  [{item.score:.4f}] {item.value['text'][:50]}...")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
Semantic Search Configuration:
  InMemoryStore(index={
      "dims": 1536,
      "embed": "openai:text-embedding-3-small"
  })

Search API:
  store.search(namespace, query=..., limit=..., filter=...)
  - Returns items with similarity scores
  - Higher score = more relevant
  - filter: metadata-based filtering

Key Fields:
  - Store 'text' field for semantic indexing
  - Additional fields for metadata/filtering

Score Interpretation:
  - Cosine similarity (0 to 1)
  - > 0.8: High relevance
  - 0.5-0.8: Medium relevance
  - < 0.5: Low relevance

Comparison with CrewAI:
  - CrewAI: ChromaDB with automatic embedding
  - LangGraph: Explicit configuration, same underlying approach
  - Both use cosine similarity search
""")


if __name__ == "__main__":
    main()
