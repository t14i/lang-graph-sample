"""
Durable Execution - Part 1: Basic Checkpoint Behavior
When is checkpoint saved? What's stored? Resume after restart?
"""

import os
import sys
import sqlite3
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

DB_PATH = "checkpoints.db"

# Global connection for persistence across graph instances
_conn = None

def get_connection():
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return _conn

def reset_connection():
    global _conn
    if _conn:
        _conn.close()
    _conn = None
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)


# =============================================================================
# State with custom fields
# =============================================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]
    step_count: int
    metadata: dict


# =============================================================================
# Nodes that track execution
# =============================================================================

llm = ChatAnthropic(model="claude-sonnet-4-20250514")


def step1(state: State) -> State:
    print(f"  [Step1] Executing... (current step_count: {state.get('step_count', 0)})")
    response = llm.invoke(state["messages"])
    return {
        "messages": [response],
        "step_count": state.get("step_count", 0) + 1,
        "metadata": {**state.get("metadata", {}), "step1_done": True}
    }


def step2(state: State) -> State:
    print(f"  [Step2] Executing... (current step_count: {state.get('step_count', 0)})")
    response = llm.invoke(state["messages"] + [HumanMessage(content="Continue with more detail.")])
    return {
        "messages": [response],
        "step_count": state.get("step_count", 0) + 1,
        "metadata": {**state.get("metadata", {}), "step2_done": True}
    }


def step3(state: State) -> State:
    print(f"  [Step3] Executing... (current step_count: {state.get('step_count', 0)})")
    response = llm.invoke(state["messages"] + [HumanMessage(content="Summarize in one sentence.")])
    return {
        "messages": [response],
        "step_count": state.get("step_count", 0) + 1,
        "metadata": {**state.get("metadata", {}), "step3_done": True}
    }


# =============================================================================
# Graph
# =============================================================================

def build_graph():
    builder = StateGraph(State)
    builder.add_node("step1", step1)
    builder.add_node("step2", step2)
    builder.add_node("step3", step3)

    builder.add_edge(START, "step1")
    builder.add_edge("step1", "step2")
    builder.add_edge("step2", "step3")
    builder.add_edge("step3", END)

    # SQLite checkpointer for durability
    checkpointer = SqliteSaver(get_connection())
    return builder.compile(checkpointer=checkpointer)


# =============================================================================
# Tests
# =============================================================================

def test_checkpoint_timing():
    """Test when checkpoints are saved."""
    print("\n" + "="*60)
    print("TEST: Checkpoint Timing")
    print("="*60)

    # Clean start
    reset_connection()

    graph = build_graph()
    config = {"configurable": {"thread_id": "timing-test"}}

    print("\nExecuting graph...")
    result = graph.invoke(
        {
            "messages": [("user", "Explain Python in 3 points")],
            "step_count": 0,
            "metadata": {}
        },
        config=config
    )

    print(f"\nFinal step_count: {result['step_count']}")
    print(f"Final metadata: {result['metadata']}")

    # Check checkpoint history
    print("\nCheckpoint history:")
    for i, state in enumerate(graph.get_state_history(config)):
        print(f"  [{i}] next={state.next}, step_count={state.values.get('step_count', 'N/A')}")
        if i > 5:
            print("  ... (truncated)")
            break


def test_resume_after_interrupt():
    """Simulate process restart by creating new graph instance."""
    print("\n" + "="*60)
    print("TEST: Resume After Restart")
    print("="*60)

    # Clean start
    reset_connection()

    thread_id = "restart-test"
    config = {"configurable": {"thread_id": thread_id}}

    # First execution - will complete step1 only
    print("\n[Phase 1] First execution...")
    graph1 = build_graph()

    # Use stream to stop after step1
    for chunk in graph1.stream(
        {
            "messages": [("user", "Explain LangGraph briefly")],
            "step_count": 0,
            "metadata": {}
        },
        config=config,
        stream_mode="values"
    ):
        if chunk.get("metadata", {}).get("step1_done"):
            print("  Step1 completed, simulating crash...")
            break

    # Check state
    state = graph1.get_state(config)
    print(f"  State after crash: next={state.next}, step_count={state.values.get('step_count')}")

    # Simulate restart - new graph instance, same DB
    print("\n[Phase 2] After restart (new graph instance)...")
    graph2 = build_graph()

    # Check if state is preserved
    state2 = graph2.get_state(config)
    print(f"  Recovered state: next={state2.next}, step_count={state2.values.get('step_count')}")
    print(f"  Recovered metadata: {state2.values.get('metadata')}")

    # Resume execution
    if state2.next:
        print(f"\n[Phase 3] Resuming from {state2.next}...")
        result = graph2.invoke(None, config=config)
        print(f"  Final step_count: {result['step_count']}")
        print(f"  Final metadata: {result['metadata']}")
    else:
        print("  Already completed, no resume needed")


def test_state_contents():
    """Examine what's actually stored in checkpoint."""
    print("\n" + "="*60)
    print("TEST: State Contents")
    print("="*60)

    reset_connection()

    graph = build_graph()
    config = {"configurable": {"thread_id": "contents-test"}}

    result = graph.invoke(
        {
            "messages": [("user", "Hello")],
            "step_count": 0,
            "metadata": {"custom_field": "custom_value", "nested": {"a": 1, "b": 2}}
        },
        config=config
    )

    state = graph.get_state(config)
    print(f"\nStored state keys: {list(state.values.keys())}")
    print(f"Messages count: {len(state.values.get('messages', []))}")
    print(f"Custom step_count: {state.values.get('step_count')}")
    print(f"Custom metadata: {state.values.get('metadata')}")

    # Check DB size
    if os.path.exists(DB_PATH):
        size = os.path.getsize(DB_PATH)
        print(f"\nCheckpoint DB size: {size / 1024:.1f} KB")


def show_summary():
    print("\n" + "="*60)
    print("CHECKPOINT BEHAVIOR SUMMARY")
    print("="*60)
    print("""
1. Checkpoint Timing:
   - Saved AFTER each node completes
   - Every state transition is persisted
   - Full state snapshot (not diff)

2. What's Stored:
   - All State fields (messages, custom fields)
   - Nested objects preserved
   - Message history grows with each step

3. Resume Capability:
   - New graph instance can resume
   - Uses thread_id to identify conversation
   - invoke(None, config) continues from last checkpoint

4. Observations:
   - Checkpoint size grows with message history
   - Each node = 1 checkpoint
   - No automatic cleanup of old checkpoints
""")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "timing":
        test_checkpoint_timing()
    elif len(sys.argv) > 1 and sys.argv[1] == "restart":
        test_resume_after_interrupt()
    elif len(sys.argv) > 1 and sys.argv[1] == "contents":
        test_state_contents()
    else:
        test_checkpoint_timing()
        test_resume_after_interrupt()
        test_state_contents()
        show_summary()
