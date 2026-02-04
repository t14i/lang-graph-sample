"""
LangGraph Memory Verification - Part 3: Cross-Thread Persistence

Goal: Verify memory sharing across different thread_ids
- Store persists independently of thread_id
- Namespace-based user isolation
- Simulating multiple sessions

Reference: https://langchain-ai.github.io/langgraph/concepts/memory/
"""

import os
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


def main():
    print("=" * 60)
    print("LangGraph Memory - Cross-Thread Persistence")
    print("=" * 60)

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        return
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set (needed for embeddings)")
        return

    # =========================================================================
    # Setup: Store and Checkpointer (separate systems)
    # =========================================================================
    print("\n" + "-" * 40)
    print("Setup: Store vs Checkpointer")
    print("-" * 40)

    # Store: Long-term memory (cross-thread)
    store = InMemoryStore(
        index={
            "dims": 1536,
            "embed": "openai:text-embedding-3-small",
        }
    )

    # Checkpointer: Short-term memory (per-thread)
    checkpointer = MemorySaver()

    print("""
Two separate memory systems:
  1. Checkpointer (MemorySaver)
     - Per thread_id
     - Conversation history
     - Graph state for HITL

  2. Store (InMemoryStore)
     - Cross-thread
     - Long-term user preferences
     - Semantic search capable
""")

    # =========================================================================
    # Build a simple graph with store access
    # =========================================================================
    llm = ChatAnthropic(model="claude-sonnet-4-20250514")

    def agent_node(state: State, config: RunnableConfig, *, store: BaseStore):
        """Agent that can read/write to store."""
        user_id = config.get("configurable", {}).get("user_id", "anonymous")
        messages = state["messages"]
        last_message = messages[-1].content if messages else ""

        # Check for memory-related commands
        if "remember:" in last_message.lower():
            # Extract what to remember (case-insensitive)
            lower_msg = last_message.lower()
            idx = lower_msg.find("remember:")
            memory_text = last_message[idx + len("remember:"):].strip()
            memory_key = f"memory_{len(list(store.search(('users', user_id))))}"
            store.put(("users", user_id), memory_key, {"text": memory_text})
            response = f"I'll remember that: '{memory_text}'"

        elif "what do you remember" in last_message.lower():
            # Retrieve all memories
            memories = list(store.search(("users", user_id)))
            if memories:
                memory_texts = [m.value.get("text", str(m.value)) for m in memories]
                response = f"I remember {len(memories)} things about you:\n" + "\n".join(f"- {t}" for t in memory_texts)
            else:
                response = "I don't have any memories about you yet."

        elif "search:" in last_message.lower():
            # Semantic search (case-insensitive)
            lower_msg = last_message.lower()
            idx = lower_msg.find("search:")
            query = last_message[idx + len("search:"):].strip()
            results = list(store.search(("users", user_id), query=query, limit=3))
            if results:
                response = f"Found {len(results)} relevant memories:\n"
                for r in results:
                    response += f"- [{r.score:.3f}] {r.value.get('text', str(r.value))}\n"
            else:
                response = "No relevant memories found."

        else:
            # Normal response with LLM
            llm_response = llm.invoke(messages)
            response = llm_response.content

        return {"messages": [{"role": "assistant", "content": response}]}

    # Build graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("agent", agent_node)
    graph_builder.add_edge(START, "agent")
    graph_builder.add_edge("agent", END)

    graph = graph_builder.compile(checkpointer=checkpointer, store=store)

    # =========================================================================
    # Test 1: Session 1 - Save memories
    # =========================================================================
    print("\n" + "-" * 40)
    print("Test 1: Session 1 (thread-1) - Save memories")
    print("-" * 40)

    config1 = {
        "configurable": {
            "thread_id": "thread-1",
            "user_id": "user_alice"
        }
    }

    # Save first memory
    result = graph.invoke(
        {
            "messages": [{"role": "user", "content": "Remember: I love hiking and outdoor activities"}],
            },
        config=config1
    )
    print(f"User: Remember: I love hiking and outdoor activities")
    print(f"Agent: {result['messages'][-1].content}")

    # Save second memory
    result = graph.invoke(
        {
            "messages": [{"role": "user", "content": "Remember: My favorite food is sushi"}],
            },
        config=config1
    )
    print(f"\nUser: Remember: My favorite food is sushi")
    print(f"Agent: {result['messages'][-1].content}")

    # =========================================================================
    # Test 2: Session 2 - Different thread, same user
    # =========================================================================
    print("\n" + "-" * 40)
    print("Test 2: Session 2 (thread-2) - Different thread, same user")
    print("-" * 40)

    config2 = {
        "configurable": {
            "thread_id": "thread-2",  # Different thread!
            "user_id": "user_alice"   # Same user
        }
    }

    result = graph.invoke(
        {"messages": [{"role": "user", "content": "What do you remember about me?"}]},
        config=config2
    )
    print(f"User: What do you remember about me?")
    print(f"Agent: {result['messages'][-1].content}")

    # =========================================================================
    # Test 3: Semantic search across threads
    # =========================================================================
    print("\n" + "-" * 40)
    print("Test 3: Semantic search from thread-2")
    print("-" * 40)

    result = graph.invoke(
        {"messages": [{"role": "user", "content": "Search: outdoor activities"}]},
        config=config2
    )
    print(f"User: Search: outdoor activities")
    print(f"Agent: {result['messages'][-1].content}")

    # =========================================================================
    # Test 4: Different user isolation
    # =========================================================================
    print("\n" + "-" * 40)
    print("Test 4: Different user (user_bob) - Isolation check")
    print("-" * 40)

    config_bob = {
        "configurable": {
            "thread_id": "thread-3",
            "user_id": "user_bob"
        }
    }

    result = graph.invoke(
        {"messages": [{"role": "user", "content": "What do you remember about me?"}]},
        config=config_bob
    )
    print(f"User (Bob): What do you remember about me?")
    print(f"Agent: {result['messages'][-1].content}")

    # Save Bob's memory
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "Remember: I work as a data scientist"}]},
        config=config_bob
    )
    print(f"\nUser (Bob): Remember: I work as a data scientist")
    print(f"Agent: {result['messages'][-1].content}")

    # =========================================================================
    # Test 5: Verify isolation - Alice's memories unchanged
    # =========================================================================
    print("\n" + "-" * 40)
    print("Test 5: Verify Alice's memories unchanged")
    print("-" * 40)

    config_alice_new = {
        "configurable": {
            "thread_id": "thread-4",  # Yet another thread
            "user_id": "user_alice"
        }
    }

    result = graph.invoke(
        {"messages": [{"role": "user", "content": "What do you remember about me?"}]},
        config=config_alice_new
    )
    print(f"User (Alice, new session): What do you remember about me?")
    print(f"Agent: {result['messages'][-1].content}")

    # =========================================================================
    # Direct store inspection
    # =========================================================================
    print("\n" + "-" * 40)
    print("Direct store inspection")
    print("-" * 40)

    print("\nAlice's memories:")
    for item in store.search(("users", "user_alice")):
        print(f"  - {item.key}: {item.value}")

    print("\nBob's memories:")
    for item in store.search(("users", "user_bob")):
        print(f"  - {item.key}: {item.value}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
Cross-Thread Memory Architecture:

  ┌─────────────────────────────────────────────────────┐
  │                    Store (Long-term)                │
  │  ┌─────────────────┐    ┌─────────────────┐        │
  │  │  users/alice/   │    │  users/bob/     │        │
  │  │  - memory_0     │    │  - memory_0     │        │
  │  │  - memory_1     │    │                 │        │
  │  └─────────────────┘    └─────────────────┘        │
  └─────────────────────────────────────────────────────┘
                    ↑ shared across all threads

  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ thread-1 │  │ thread-2 │  │ thread-3 │  │ thread-4 │
  │ (Alice)  │  │ (Alice)  │  │  (Bob)   │  │ (Alice)  │
  └──────────┘  └──────────┘  └──────────┘  └──────────┘
        ↓              ↓              ↓              ↓
  ┌─────────────────────────────────────────────────────┐
  │              Checkpointer (Short-term)              │
  │       Each thread has separate conversation         │
  └─────────────────────────────────────────────────────┘

Key Points:
  - Store: Shared across all threads, namespace isolates users
  - Checkpointer: Per-thread conversation history
  - user_id in namespace enables user isolation
  - Semantic search works across threads for same user

Comparison with CrewAI:
  - CrewAI: Long-term memory automatic with memory=True
  - LangGraph: Explicit Store API, more control
  - Both support cross-session persistence
""")


if __name__ == "__main__":
    main()
