"""
LangGraph Memory Verification - Part 4: LangMem Memory Tools

Goal: Verify LangMem tools for agent-managed memory
- create_manage_memory_tool: Agent saves/updates/deletes memories
- create_search_memory_tool: Agent searches memories
- Agent autonomously decides when to use memory

Reference: https://langchain-ai.github.io/langmem/
"""

import os
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import create_react_agent
from langmem import create_manage_memory_tool, create_search_memory_tool


def main():
    print("=" * 60)
    print("LangGraph Memory - LangMem Memory Tools")
    print("=" * 60)

    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        return

    # =========================================================================
    # Setup: Store with semantic search
    # =========================================================================
    print("\n" + "-" * 40)
    print("Setup: Store and Memory Tools")
    print("-" * 40)

    store = InMemoryStore(
        index={
            "dims": 1536,
            "embed": "openai:text-embedding-3-small",
        }
    )

    # Create memory tools with namespace template
    # {user_id} will be replaced from config
    manage_memory = create_manage_memory_tool(namespace=("memories", "{user_id}"))
    search_memory = create_search_memory_tool(namespace=("memories", "{user_id}"))

    print("Created tools:")
    print(f"  - {manage_memory.name}: {manage_memory.description[:80]}...")
    print(f"  - {search_memory.name}: {search_memory.description[:80]}...")

    # =========================================================================
    # Create ReAct agent with memory tools
    # =========================================================================
    print("\n" + "-" * 40)
    print("Creating ReAct Agent with Memory Tools")
    print("-" * 40)

    agent = create_react_agent(
        "openai:gpt-4o",
        tools=[manage_memory, search_memory],
        store=store,
    )

    print("Agent created with gpt-4o model and memory tools")

    # =========================================================================
    # Test 1: Agent saves memory autonomously
    # =========================================================================
    print("\n" + "-" * 40)
    print("Test 1: Tell agent personal info - will it save?")
    print("-" * 40)

    config = {"configurable": {"user_id": "user_123", "thread_id": "test-1"}}

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "My name is Taro Tanaka and I'm a software engineer. Please remember this."}]},
        config=config
    )

    print(f"User: My name is Taro Tanaka and I'm a software engineer. Please remember this.")
    print(f"Agent: {response['messages'][-1].content}")

    # Check what was stored
    print("\n[Store contents after Test 1]")
    memories = list(store.search(("memories", "user_123")))
    for m in memories:
        print(f"  - {m.key}: {m.value}")

    # =========================================================================
    # Test 2: Agent searches memory
    # =========================================================================
    print("\n" + "-" * 40)
    print("Test 2: Ask about previously stored info")
    print("-" * 40)

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "What's my name and what do I do?"}]},
        config={"configurable": {"user_id": "user_123", "thread_id": "test-2"}}
    )

    print(f"User: What's my name and what do I do?")
    print(f"Agent: {response['messages'][-1].content}")

    # =========================================================================
    # Test 3: Add more memories
    # =========================================================================
    print("\n" + "-" * 40)
    print("Test 3: Add more personal info")
    print("-" * 40)

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "I love hiking, especially in the mountains. My favorite food is ramen. Please save these preferences."}]},
        config={"configurable": {"user_id": "user_123", "thread_id": "test-3"}}
    )

    print(f"User: I love hiking, especially in the mountains. My favorite food is ramen. Please save these preferences.")
    print(f"Agent: {response['messages'][-1].content}")

    # Check stored memories
    print("\n[Store contents after Test 3]")
    memories = list(store.search(("memories", "user_123")))
    for m in memories:
        print(f"  - {m.key}: {m.value}")

    # =========================================================================
    # Test 4: Semantic search by agent
    # =========================================================================
    print("\n" + "-" * 40)
    print("Test 4: Ask about food preferences (semantic search)")
    print("-" * 40)

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "What kind of food do I like?"}]},
        config={"configurable": {"user_id": "user_123", "thread_id": "test-4"}}
    )

    print(f"User: What kind of food do I like?")
    print(f"Agent: {response['messages'][-1].content}")

    # =========================================================================
    # Test 5: Update existing memory
    # =========================================================================
    print("\n" + "-" * 40)
    print("Test 5: Update preference")
    print("-" * 40)

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "Actually, I changed my mind. My favorite food is now sushi, not ramen. Please update my preferences."}]},
        config={"configurable": {"user_id": "user_123", "thread_id": "test-5"}}
    )

    print(f"User: Actually, I changed my mind. My favorite food is now sushi, not ramen. Please update my preferences.")
    print(f"Agent: {response['messages'][-1].content}")

    # Check updated memories
    print("\n[Store contents after update]")
    memories = list(store.search(("memories", "user_123")))
    for m in memories:
        print(f"  - {m.key}: {m.value}")

    # =========================================================================
    # Test 6: Different user isolation
    # =========================================================================
    print("\n" + "-" * 40)
    print("Test 6: Different user (user_456) - check isolation")
    print("-" * 40)

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "Do you know my name or any of my preferences?"}]},
        config={"configurable": {"user_id": "user_456", "thread_id": "test-6"}}
    )

    print(f"User (user_456): Do you know my name or any of my preferences?")
    print(f"Agent: {response['messages'][-1].content}")

    # =========================================================================
    # Test 7: Check tool usage in messages
    # =========================================================================
    print("\n" + "-" * 40)
    print("Test 7: Examine agent's tool usage")
    print("-" * 40)

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "Tell me everything you know about me from memory."}]},
        config={"configurable": {"user_id": "user_123", "thread_id": "test-7"}}
    )

    print(f"User: Tell me everything you know about me from memory.")
    print(f"\nMessages in response:")
    for msg in response['messages']:
        msg_type = type(msg).__name__
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            print(f"  [{msg_type}] tool_calls: {[tc['name'] for tc in msg.tool_calls]}")
        elif hasattr(msg, 'name'):
            print(f"  [{msg_type}] tool: {msg.name}, content: {str(msg.content)[:100]}...")
        else:
            content_preview = str(msg.content)[:100] if msg.content else "(empty)"
            print(f"  [{msg_type}] {content_preview}...")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
LangMem Memory Tools:

  create_manage_memory_tool(namespace=(...))
    - Agent can save, update, delete memories
    - Namespace template: {user_id} replaced from config

  create_search_memory_tool(namespace=(...))
    - Agent can search memories semantically
    - Returns relevant memories based on query

Agent Behavior:
  - Autonomously decides when to save/search
  - Responds to explicit "remember" requests
  - Searches when asked about past information
  - Updates memories when preferences change

Key Benefits:
  - Agent manages its own memory
  - No manual memory management needed
  - Semantic search for relevance
  - User isolation via namespace

Comparison with CrewAI:
  - CrewAI: Memory is implicit (memory=True)
  - LangGraph + LangMem: Agent has explicit tools
  - LangMem: More control over what gets stored
  - LangMem: Agent can update/delete memories (CrewAI cannot)

This is a unique feature not available in CrewAI!
""")


if __name__ == "__main__":
    main()
