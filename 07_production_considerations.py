"""
LangGraph HITL - Production Considerations
What's missing for production use?
"""

# =============================================================================
# PRODUCTION CONSIDERATIONS FOR LANGGRAPH HITL
# =============================================================================

"""
## 1. CHECKPOINTER - State Persistence

Development:
    from langgraph.checkpoint.memory import MemorySaver
    checkpointer = MemorySaver()  # In-memory, lost on restart

Production options:
    # PostgreSQL (recommended for production)
    from langgraph_checkpoint_postgres import PostgresSaver
    checkpointer = PostgresSaver.from_conn_string("postgresql://...")

    # SQLite (for simpler deployments)
    from langgraph.checkpoint.sqlite import SqliteSaver
    checkpointer = SqliteSaver.from_conn_string("sqlite:///checkpoints.db")

VERDICT: ✅ Good support, just need to configure


## 2. AUDIT LOGGING - Who approved what, when?

LangGraph does NOT provide built-in audit logging.
You need to implement it yourself:

    def human_approval(state: State) -> Command:
        decision = interrupt({...})

        # Manual audit logging
        audit_log.record(
            timestamp=datetime.now(),
            user_id=get_current_user(),  # Where does this come from?
            action=decision["action"],
            tool_name=tool_call["name"],
            tool_args=tool_call["args"],
        )

VERDICT: ⚠️ Must implement yourself, user context not built-in


## 3. TIMEOUT - What if approval never comes?

LangGraph has NO built-in timeout for interrupts.
Options:
    1. Background job to check stale threads
    2. Application-level timeout when polling for state
    3. Scheduled cleanup of abandoned threads

    # Example: Check for stale threads
    for thread_id in get_active_threads():
        state = graph.get_state({"configurable": {"thread_id": thread_id}})
        if state.next and is_stale(state.created_at, timeout=timedelta(hours=24)):
            # Auto-reject or notify
            graph.invoke(
                Command(resume={"action": "reject", "reason": "Timeout"}),
                config={"configurable": {"thread_id": thread_id}}
            )

VERDICT: ⚠️ Must implement yourself


## 4. NOTIFICATION - How to notify approvers?

LangGraph has NO notification system.
You need to:
    1. Detect interrupt (check state.next)
    2. Send notification (email, Slack, webhook)
    3. Provide approval UI/endpoint

VERDICT: ⚠️ Must implement yourself


## 5. AUTHORIZATION - Who can approve?

LangGraph has NO built-in authorization.
You need to:
    1. Track which user initiated the request
    2. Define approval policies (who can approve what)
    3. Validate approver in human_approval node

VERDICT: ⚠️ Must implement yourself


## 6. MULTIPLE APPROVALS - Sequential/Parallel approvals

LangGraph supports multiple interrupts naturally:

    def multi_approval(state: State) -> Command:
        # First approval
        approval1 = interrupt({"stage": "manager", ...})

        # Second approval (only reached after first resume)
        approval2 = interrupt({"stage": "security", ...})

        if approval1["approved"] and approval2["approved"]:
            return Command(goto="execute")

    # Resume flow:
    # invoke() -> interrupted at manager
    # invoke(Command(resume=manager_approval)) -> interrupted at security
    # invoke(Command(resume=security_approval)) -> executes

VERDICT: ✅ Works, but each approval requires separate invoke


## 7. EDITING TOOL CALLS - Modifying args before execution

Demonstrated in 04_hitl_reject_edit.py - works well.

VERDICT: ✅ Supported


## 8. CANCELLATION - Abandoning a flow

    # Simply don't resume, or:
    graph.update_state(
        config,
        {"messages": [SystemMessage("Operation cancelled")]},
        as_node="__end__"  # Skip to end
    )

VERDICT: ✅ Possible but awkward API


## 9. RESUMABILITY ACROSS RESTARTS

With persistent checkpointer (Postgres/SQLite):
    - Server can restart
    - Resume from any thread_id
    - State fully preserved

VERDICT: ✅ Good with persistent checkpointer


## 10. STREAMING DURING APPROVAL WAIT

Can use streaming to get partial results:

    for chunk in graph.stream(input, config):
        if "__interrupt__" in chunk:
            # Handle interrupt
            pass

VERDICT: ✅ Supported

"""

# =============================================================================
# SUMMARY
# =============================================================================

SUMMARY = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ LANGGRAPH HITL - PRODUCTION READINESS SUMMARY                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ ✅ WORKS WELL:                                                              │
│    - interrupt() / Command(resume=...) API is clean                         │
│    - Approve / Reject / Edit patterns supported                             │
│    - Persistent checkpointer (Postgres/SQLite)                              │
│    - Multiple sequential approvals                                          │
│    - State resumability across restarts                                     │
│    - Streaming support                                                      │
│                                                                             │
│ ⚠️ MUST IMPLEMENT YOURSELF:                                                 │
│    - Audit logging (who approved, when)                                     │
│    - Timeout handling (stale approval requests)                             │
│    - Notification system (email/Slack/webhook)                              │
│    - Authorization (who can approve what)                                   │
│    - User context tracking                                                  │
│                                                                             │
│ 📝 VERDICT:                                                                 │
│    LangGraph provides the core HITL primitives well.                        │
│    But for production, you need to build:                                   │
│    - Approval management layer (UI, API, notifications)                     │
│    - Audit/compliance layer                                                 │
│    - Timeout/cleanup jobs                                                   │
│                                                                             │
│    It's a "bring your own approval infrastructure" situation.               │
│    The graph execution part is solid, the human workflow part is DIY.       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""

if __name__ == "__main__":
    print(SUMMARY)
