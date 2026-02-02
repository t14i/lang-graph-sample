"""
Durable Execution - Part 3: Production Considerations
Concurrent execution, cleanup, state migration, checkpoint size
"""

import os
import sys
import sqlite3
import threading
import time
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, AIMessage

DB_PATH = "checkpoints_prod.db"

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
# State
# =============================================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]
    counter: int


# =============================================================================
# Simple nodes for testing
# =============================================================================

def increment(state: State) -> State:
    time.sleep(0.1)  # Simulate work
    return {
        "messages": [AIMessage(content=f"Counter: {state['counter'] + 1}")],
        "counter": state["counter"] + 1
    }


def build_graph():
    builder = StateGraph(State)
    builder.add_node("increment", increment)
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)

    checkpointer = SqliteSaver(get_connection())
    return builder.compile(checkpointer=checkpointer)


# =============================================================================
# Test: Concurrent execution on same thread_id
# =============================================================================

def test_concurrent_same_thread():
    """What happens with concurrent invoke() on same thread_id?"""
    print("\n" + "="*60)
    print("TEST: Concurrent Execution (Same thread_id)")
    print("="*60)

    reset_connection()

    thread_id = "concurrent-test"
    config = {"configurable": {"thread_id": thread_id}}

    results = []
    errors = []

    def run_invoke(run_id):
        try:
            graph = build_graph()
            result = graph.invoke(
                {"messages": [HumanMessage(content=f"Run {run_id}")], "counter": 0},
                config=config
            )
            results.append((run_id, result["counter"]))
        except Exception as e:
            errors.append((run_id, str(e)))

    # Start multiple threads
    threads = []
    for i in range(3):
        t = threading.Thread(target=run_invoke, args=(i,))
        threads.append(t)

    print("\nStarting 3 concurrent invocations on same thread_id...")
    for t in threads:
        t.start()

    for t in threads:
        t.join()

    print(f"\nResults: {results}")
    print(f"Errors: {errors}")

    # Check final state
    graph = build_graph()
    state = graph.get_state(config)
    print(f"Final counter: {state.values.get('counter')}")
    print(f"Message count: {len(state.values.get('messages', []))}")


# =============================================================================
# Test: Checkpoint size growth
# =============================================================================

def test_checkpoint_size_growth():
    """How does checkpoint size grow with messages?"""
    print("\n" + "="*60)
    print("TEST: Checkpoint Size Growth")
    print("="*60)

    reset_connection()

    graph = build_graph()

    sizes = []
    for i in range(10):
        config = {"configurable": {"thread_id": f"size-test-{i}"}}

        # Create messages of increasing size
        messages = [HumanMessage(content=f"Message {j}: " + "x" * 100) for j in range(i + 1)]

        graph.invoke(
            {"messages": messages, "counter": 0},
            config=config
        )

        size = os.path.getsize(DB_PATH)
        sizes.append((i + 1, size))
        print(f"  {i + 1} threads, {i + 1} msgs each: {size / 1024:.1f} KB")

    print(f"\nGrowth pattern: {[(s[0], f'{s[1]/1024:.1f}KB') for s in sizes]}")


# =============================================================================
# Test: Checkpoint history and cleanup
# =============================================================================

def test_checkpoint_history():
    """Examine checkpoint history and discuss cleanup."""
    print("\n" + "="*60)
    print("TEST: Checkpoint History")
    print("="*60)

    reset_connection()

    # Build a multi-step graph
    def step1(state: State) -> State:
        return {"messages": [AIMessage(content="Step 1")], "counter": state["counter"] + 1}

    def step2(state: State) -> State:
        return {"messages": [AIMessage(content="Step 2")], "counter": state["counter"] + 1}

    def step3(state: State) -> State:
        return {"messages": [AIMessage(content="Step 3")], "counter": state["counter"] + 1}

    builder = StateGraph(State)
    builder.add_node("step1", step1)
    builder.add_node("step2", step2)
    builder.add_node("step3", step3)
    builder.add_edge(START, "step1")
    builder.add_edge("step1", "step2")
    builder.add_edge("step2", "step3")
    builder.add_edge("step3", END)

    checkpointer = SqliteSaver(get_connection())
    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "history-test"}}

    graph.invoke(
        {"messages": [HumanMessage(content="Start")], "counter": 0},
        config=config
    )

    print("\nCheckpoint history (most recent first):")
    history = list(graph.get_state_history(config))
    for i, state in enumerate(history):
        print(f"  [{i}] counter={state.values.get('counter')}, next={state.next}")

    print(f"\nTotal checkpoints: {len(history)}")
    print(f"DB size: {os.path.getsize(DB_PATH) / 1024:.1f} KB")

    print("""
Note: LangGraph does NOT auto-cleanup old checkpoints.
For production, you need to:
1. Periodically delete old checkpoints
2. Or use checkpoint_id to restore specific points
3. Monitor DB size growth
""")


# =============================================================================
# Test: Thread listing (what's possible?)
# =============================================================================

def test_thread_listing():
    """Can we list all threads? (Needed for cleanup, monitoring)"""
    print("\n" + "="*60)
    print("TEST: Thread Listing")
    print("="*60)

    reset_connection()

    graph = build_graph()

    # Create multiple threads
    for i in range(5):
        config = {"configurable": {"thread_id": f"list-test-{i}"}}
        graph.invoke(
            {"messages": [HumanMessage(content=f"Thread {i}")], "counter": i},
            config=config
        )

    # Try to list threads - this is checkpointer-dependent
    print("\nAttempting to list threads...")

    # SqliteSaver doesn't have a direct list_threads method
    # We need to query the DB directly
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check table structure
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"Tables: {[t[0] for t in tables]}")

    # Try to get unique thread_ids
    try:
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
        threads = cursor.fetchall()
        print(f"Thread IDs: {[t[0] for t in threads]}")
    except Exception as e:
        print(f"Could not query threads: {e}")

    conn.close()

    print("""
Observation:
- No built-in API to list all thread_ids
- Must query checkpointer storage directly
- Postgres/SQLite have different schemas
- Need custom cleanup jobs
""")


# =============================================================================
# Summary
# =============================================================================

def show_summary():
    print("\n" + "="*60)
    print("PRODUCTION CONSIDERATIONS SUMMARY")
    print("="*60)
    print("""
1. CONCURRENT EXECUTION:
   - Same thread_id: Last write wins, may cause inconsistency
   - Different thread_ids: Safe, fully isolated
   - Recommendation: Generate unique thread_id per conversation

2. CHECKPOINT SIZE:
   - Grows with each node execution
   - Full state snapshot (not diff)
   - Message history accumulates
   - Monitor and set limits

3. CLEANUP:
   - No auto-cleanup of old checkpoints
   - No built-in retention policy
   - Must implement custom cleanup job
   - Query checkpointer storage directly

4. THREAD MANAGEMENT:
   - No API to list all threads
   - Need to track thread_ids externally
   - Or query storage directly (DB-specific)

5. STATE MIGRATION:
   - Adding new fields: OK (default values)
   - Removing fields: May break on resume
   - Changing types: Dangerous
   - Recommendation: Version your state schema

6. PRODUCTION CHECKLIST:
   [ ] Unique thread_id generation
   [ ] Checkpoint size monitoring
   [ ] Cleanup job for old checkpoints
   [ ] External thread tracking
   [ ] State schema versioning
   [ ] DB connection pooling
   [ ] Backup strategy
""")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "concurrent":
        test_concurrent_same_thread()
    elif len(sys.argv) > 1 and sys.argv[1] == "size":
        test_checkpoint_size_growth()
    elif len(sys.argv) > 1 and sys.argv[1] == "history":
        test_checkpoint_history()
    elif len(sys.argv) > 1 and sys.argv[1] == "threads":
        test_thread_listing()
    else:
        test_concurrent_same_thread()
        test_checkpoint_size_growth()
        test_checkpoint_history()
        test_thread_listing()
        show_summary()
