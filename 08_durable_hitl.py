"""
Durable Execution - Part 2: HITL + Durability
interrupt() -> process restart -> Command(resume=...) -> continue
"""

import os
import sys
import sqlite3
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

DB_PATH = "checkpoints_hitl.db"

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
    approval_count: int


# =============================================================================
# Tools
# =============================================================================

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {to}"


@tool
def delete_record(record_id: str) -> str:
    """Delete a database record."""
    return f"Record {record_id} deleted"


tools = [send_email, delete_record]


# =============================================================================
# Nodes
# =============================================================================

llm = ChatAnthropic(model="claude-sonnet-4-20250514")
llm_with_tools = llm.bind_tools(tools)


def agent(state: State) -> State:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response], "approval_count": state.get("approval_count", 0)}


def human_approval(state: State) -> Command:
    """Interrupt for human approval - survives restart."""
    last_message = state["messages"][-1]

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return Command(goto=END)

    tool_call = last_message.tool_calls[0]

    # This interrupt survives process restart
    decision = interrupt({
        "action": "approve_tool_call",
        "tool_name": tool_call["name"],
        "tool_args": tool_call["args"],
        "approval_count": state.get("approval_count", 0),
    })

    if decision.get("action") == "approve":
        return Command(
            goto="tools",
            update={"approval_count": state.get("approval_count", 0) + 1}
        )
    elif decision.get("action") == "reject":
        msg = ToolMessage(
            content=f"Rejected: {decision.get('reason', 'No reason')}",
            tool_call_id=tool_call["id"]
        )
        return Command(goto="agent", update={"messages": [msg]})

    return Command(goto=END)


def should_continue(state: State) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "human_approval"
    return END


# =============================================================================
# Graph
# =============================================================================

def build_graph():
    builder = StateGraph(State)
    builder.add_node("agent", agent)
    builder.add_node("human_approval", human_approval)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, ["human_approval", END])
    builder.add_edge("tools", "agent")

    checkpointer = SqliteSaver(get_connection())
    return builder.compile(checkpointer=checkpointer)


# =============================================================================
# Test scenarios
# =============================================================================

def test_hitl_survives_restart():
    """Test that HITL interrupt survives process restart."""
    print("\n" + "="*60)
    print("TEST: HITL Survives Restart")
    print("="*60)

    reset_connection()

    thread_id = "hitl-restart-test"
    config = {"configurable": {"thread_id": thread_id}}

    # Phase 1: Start execution, hit interrupt
    print("\n[Phase 1] Starting execution...")
    graph1 = build_graph()

    result1 = graph1.invoke(
        {
            "messages": [("user", "Send an email to test@example.com with subject 'Hello' and body 'Hi there'")],
            "approval_count": 0
        },
        config=config
    )

    state1 = graph1.get_state(config)
    print(f"  Interrupted at: {state1.next}")
    if state1.tasks:
        print(f"  Interrupt value: {state1.tasks[0].interrupts[0].value}")

    # Phase 2: Simulate restart - new graph instance
    print("\n[Phase 2] Simulating restart (new graph instance)...")
    graph2 = build_graph()

    state2 = graph2.get_state(config)
    print(f"  Recovered state: next={state2.next}")
    print(f"  Approval count: {state2.values.get('approval_count')}")

    if state2.tasks:
        print(f"  Recovered interrupt: {state2.tasks[0].interrupts[0].value}")

    # Phase 3: Resume with approval
    print("\n[Phase 3] Resuming with approval...")
    result2 = graph2.invoke(Command(resume={"action": "approve"}), config=config)

    print(f"  Final response: {result2['messages'][-1].content[:100]}...")
    print(f"  Final approval_count: {result2['approval_count']}")


def test_multiple_approvals_with_restart():
    """Test multiple approvals with restart in between."""
    print("\n" + "="*60)
    print("TEST: Multiple Approvals with Restart")
    print("="*60)

    reset_connection()

    thread_id = "multi-approval-test"
    config = {"configurable": {"thread_id": thread_id}}

    # Phase 1: First tool call
    print("\n[Phase 1] First request...")
    graph1 = build_graph()

    graph1.invoke(
        {
            "messages": [("user", "First, send email to a@example.com subject 'Test' body 'Hello'")],
            "approval_count": 0
        },
        config=config
    )

    state = graph1.get_state(config)
    print(f"  Interrupted for: {state.tasks[0].interrupts[0].value['tool_name'] if state.tasks else 'N/A'}")

    # Approve first
    print("\n[Phase 2] Approving first...")
    graph1.invoke(Command(resume={"action": "approve"}), config=config)

    # Add second request
    print("\n[Phase 3] Second request...")
    graph1.invoke(
        {"messages": [("user", "Now delete record ID 12345")]},
        config=config
    )

    state = graph1.get_state(config)
    print(f"  Interrupted for: {state.tasks[0].interrupts[0].value['tool_name'] if state.tasks else 'N/A'}")

    # Simulate restart
    print("\n[Phase 4] Restart and recover...")
    graph2 = build_graph()

    state2 = graph2.get_state(config)
    print(f"  Recovered interrupt: {state2.tasks[0].interrupts[0].value['tool_name'] if state2.tasks else 'N/A'}")
    print(f"  Approval count so far: {state2.values.get('approval_count')}")

    # Approve second
    print("\n[Phase 5] Approving second after restart...")
    result = graph2.invoke(Command(resume={"action": "approve"}), config=config)

    print(f"  Final approval_count: {result['approval_count']}")


def test_reject_after_restart():
    """Test rejection after restart."""
    print("\n" + "="*60)
    print("TEST: Reject After Restart")
    print("="*60)

    reset_connection()

    thread_id = "reject-restart-test"
    config = {"configurable": {"thread_id": thread_id}}

    # Start and interrupt
    print("\n[Phase 1] Starting...")
    graph1 = build_graph()

    graph1.invoke(
        {
            "messages": [("user", "Delete record 999")],
            "approval_count": 0
        },
        config=config
    )

    # Restart
    print("\n[Phase 2] Restart...")
    graph2 = build_graph()

    state = graph2.get_state(config)
    print(f"  Recovered interrupt: {state.tasks[0].interrupts[0].value if state.tasks else 'N/A'}")

    # Reject
    print("\n[Phase 3] Rejecting after restart...")
    result = graph2.invoke(
        Command(resume={"action": "reject", "reason": "Too dangerous"}),
        config=config
    )

    print(f"  Response after rejection: {result['messages'][-1].content[:150]}...")


def show_summary():
    print("\n" + "="*60)
    print("HITL + DURABLE EXECUTION SUMMARY")
    print("="*60)
    print("""
1. Interrupt Persistence:
   - interrupt() state survives process restart
   - Interrupt value fully preserved in checkpoint
   - Can resume with Command(resume=...) after restart

2. Approval Flow:
   - approval_count and other custom state preserved
   - Multiple sequential approvals work across restarts
   - Reject after restart works correctly

3. Key Points:
   - Same thread_id required to resume
   - New graph instance can pick up where old one stopped
   - No data loss on crash/restart

4. Production Implications:
   - Server can restart without losing pending approvals
   - Long-running approval workflows are safe
   - Need external system to track pending approvals
   - Notification on restart: check for interrupted threads
""")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "basic":
        test_hitl_survives_restart()
    elif len(sys.argv) > 1 and sys.argv[1] == "multi":
        test_multiple_approvals_with_restart()
    elif len(sys.argv) > 1 and sys.argv[1] == "reject":
        test_reject_after_restart()
    else:
        test_hitl_survives_restart()
        test_multiple_approvals_with_restart()
        test_reject_after_restart()
        show_summary()
