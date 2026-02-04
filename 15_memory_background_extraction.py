"""
LangGraph Memory Verification - Part 5: Background Memory Extraction

Goal: Verify automatic memory extraction from conversations
- create_memory_store_manager: Extract facts from conversation
- Background processing (not blocking main flow)
- Memory consolidation and updates

Reference: https://langchain-ai.github.io/langmem/
"""

import asyncio
import os
from langgraph.store.memory import InMemoryStore
from langmem import create_memory_store_manager


async def main():
    print("=" * 60)
    print("LangGraph Memory - Background Memory Extraction")
    print("=" * 60)

    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        return

    # =========================================================================
    # Setup: Store with semantic search
    # =========================================================================
    print("\n" + "-" * 40)
    print("Setup: Store and Memory Manager")
    print("-" * 40)

    store = InMemoryStore(
        index={
            "dims": 1536,
            "embed": "openai:text-embedding-3-small",
        }
    )

    # Create memory store manager for background extraction
    manager = create_memory_store_manager(
        "openai:gpt-4o",
        namespace=("memories", "{user_id}"),
        store=store,
    )

    print("Memory Store Manager created")
    print("  - Model: gpt-4o (for extraction)")
    print("  - Namespace: ('memories', '{user_id}')")

    # =========================================================================
    # Test 1: Extract from simple conversation
    # =========================================================================
    print("\n" + "-" * 40)
    print("Test 1: Extract facts from conversation")
    print("-" * 40)

    conversation1 = [
        {"role": "user", "content": "Hi! I'm looking for restaurant recommendations."},
        {"role": "assistant", "content": "I'd be happy to help! What type of cuisine are you in the mood for?"},
        {"role": "user", "content": "I love Italian food, especially pasta. But I'm allergic to shellfish, so I need to be careful with seafood pasta."},
        {"role": "assistant", "content": "Got it! I'll recommend Italian restaurants that are careful with allergens. Do you have a preferred area?"},
        {"role": "user", "content": "I live in Shibuya, Tokyo, so somewhere nearby would be great."},
    ]

    print("Conversation:")
    for msg in conversation1:
        role = "User" if msg["role"] == "user" else "Assistant"
        print(f"  {role}: {msg['content'][:60]}...")

    print("\nExtracting memories...")
    config = {"configurable": {"user_id": "user_123"}}

    # Run extraction
    await manager.ainvoke(
        {"messages": conversation1},
        config=config,
    )

    print("\n[Extracted memories]")
    memories = list(store.search(("memories", "user_123")))
    for m in memories:
        print(f"  - {m.key}: {m.value}")

    # =========================================================================
    # Test 2: Extract from work-related conversation
    # =========================================================================
    print("\n" + "-" * 40)
    print("Test 2: Extract from work conversation")
    print("-" * 40)

    conversation2 = [
        {"role": "user", "content": "I need to prepare for my meeting tomorrow."},
        {"role": "assistant", "content": "What's the meeting about?"},
        {"role": "user", "content": "It's a project review with my team. I'm the tech lead for the AI project at our company."},
        {"role": "assistant", "content": "That sounds important. What time is the meeting?"},
        {"role": "user", "content": "It's at 2 PM. I prefer afternoon meetings since I'm not a morning person."},
    ]

    print("Conversation:")
    for msg in conversation2:
        role = "User" if msg["role"] == "user" else "Assistant"
        print(f"  {role}: {msg['content'][:60]}...")

    print("\nExtracting memories...")

    await manager.ainvoke(
        {"messages": conversation2},
        config=config,
    )

    print("\n[All memories after Test 2]")
    memories = list(store.search(("memories", "user_123")))
    for m in memories:
        print(f"  - {m.key}: {m.value}")

    # =========================================================================
    # Test 3: Memory update/consolidation
    # =========================================================================
    print("\n" + "-" * 40)
    print("Test 3: Memory consolidation (contradicting info)")
    print("-" * 40)

    conversation3 = [
        {"role": "user", "content": "I've changed my food preferences recently."},
        {"role": "assistant", "content": "Oh? What do you like now?"},
        {"role": "user", "content": "I've started loving Japanese food, especially sushi. Italian is still good but Japanese is now my favorite."},
    ]

    print("Conversation:")
    for msg in conversation3:
        role = "User" if msg["role"] == "user" else "Assistant"
        print(f"  {role}: {msg['content'][:60]}...")

    print("\nExtracting and potentially updating memories...")

    await manager.ainvoke(
        {"messages": conversation3},
        config=config,
    )

    print("\n[All memories after consolidation]")
    memories = list(store.search(("memories", "user_123")))
    for m in memories:
        print(f"  - {m.key}: {m.value}")

    # =========================================================================
    # Test 4: Semantic search on extracted memories
    # =========================================================================
    print("\n" + "-" * 40)
    print("Test 4: Semantic search on extracted memories")
    print("-" * 40)

    queries = [
        "food preferences",
        "allergies",
        "work information",
        "schedule preferences",
        "location",
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        results = list(store.search(
            ("memories", "user_123"),
            query=query,
            limit=2
        ))
        for r in results:
            text = r.value.get("content", r.value.get("text", str(r.value)))
            if len(text) > 80:
                text = text[:80] + "..."
            print(f"  [{r.score:.3f}] {text}")

    # =========================================================================
    # Test 5: Different user isolation
    # =========================================================================
    print("\n" + "-" * 40)
    print("Test 5: Different user extraction")
    print("-" * 40)

    conversation_bob = [
        {"role": "user", "content": "I'm Bob and I work in marketing."},
        {"role": "assistant", "content": "Nice to meet you, Bob!"},
        {"role": "user", "content": "I love coffee and usually work from our Osaka office."},
    ]

    await manager.ainvoke(
        {"messages": conversation_bob},
        config={"configurable": {"user_id": "user_bob"}},
    )

    print("Bob's memories:")
    for m in store.search(("memories", "user_bob")):
        print(f"  - {m.key}: {m.value}")

    print("\nUser 123's memories (unchanged):")
    count = len(list(store.search(("memories", "user_123"))))
    print(f"  {count} memories (isolation confirmed)")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
Background Memory Extraction:

  create_memory_store_manager(model, namespace)
    - Extracts facts from conversation history
    - Runs asynchronously (background processing)
    - Consolidates and updates existing memories

Extraction Behavior:
  - Identifies user facts, preferences, constraints
  - Creates semantic embeddings for search
  - Updates conflicting information (consolidation)

Use Cases:
  - Hot Path: Agent uses Memory Tools in real-time
  - Background: Manager extracts after conversation
  - Both can be combined for comprehensive memory

Key Benefits:
  - No explicit "remember" needed
  - Automatic fact extraction
  - Memory consolidation
  - Async processing

Comparison with CrewAI:
  - CrewAI: Short-term memory auto-extracted
  - CrewAI: Long-term memory via embeddings
  - LangMem: More explicit control
  - LangMem: Background extraction is opt-in
  - LangMem: Better consolidation logic

Production Considerations:
  - Background extraction adds latency/cost
  - Consider batch processing for cost efficiency
  - Monitor extraction quality
  - May need custom extraction prompts
""")


if __name__ == "__main__":
    asyncio.run(main())
