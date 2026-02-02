"""
LangGraph Production Considerations
What's missing for production use? (HITL + Durable Execution)
"""

# =============================================================================
# PRODUCTION CONSIDERATIONS FOR LANGGRAPH
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


# =============================================================================
# DURABLE EXECUTION CONSIDERATIONS
# =============================================================================

## 11. CHECKPOINT CLEANUP - Old checkpoints accumulate

LangGraph does NOT auto-cleanup old checkpoints.
Each node execution creates a new checkpoint.
DB grows indefinitely.

    # Must implement cleanup job
    # Query checkpointer storage directly (DB-specific)
    cursor.execute("DELETE FROM checkpoints WHERE created_at < ?", [cutoff])

VERDICT: ⚠️ Must implement yourself


## 12. THREAD LISTING - Finding active/pending threads

No built-in API to list all thread_ids.
Need this for:
    - Finding pending approval requests
    - Cleanup jobs
    - Monitoring dashboards

    # Must query storage directly
    cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")

VERDICT: ⚠️ Must implement yourself (DB-specific)


## 13. CONCURRENT ACCESS - Same thread_id race condition

Concurrent invoke() on same thread_id causes race conditions.
Last write wins, state may be inconsistent.

    # BAD: Two requests on same thread_id
    thread1: invoke(msg1, thread_id="abc")  # counter=1
    thread2: invoke(msg2, thread_id="abc")  # counter=1 (not 2!)

    # GOOD: Unique thread_id per conversation
    thread_id = f"user-{user_id}-{uuid4()}"

VERDICT: ⚠️ Must generate unique thread_ids


## 14. STATE SCHEMA MIGRATION - Changing State definition

Adding new fields: OK (use default values)
Removing fields: May break on resume
Changing types: Dangerous

    # Version your state schema
    class StateV2(TypedDict):
        messages: Annotated[list, add_messages]
        counter: int
        new_field: str = ""  # Added in V2

VERDICT: ⚠️ Must manage schema versions carefully


## 15. CHECKPOINT SIZE - State grows with messages

Full state snapshot per checkpoint.
Message history accumulates.
Long conversations = large checkpoints.

    # Consider:
    # 1. Message summarization
    # 2. Checkpoint compression
    # 3. Max message limits

VERDICT: ⚠️ Monitor and manage


"""

# =============================================================================
# SUMMARY
# =============================================================================

SUMMARY = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ LANGGRAPH PRODUCTION READINESS SUMMARY                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ ✅ WORKS WELL:                                                              │
│    - interrupt() / Command(resume=...) API is clean                         │
│    - Approve / Reject / Edit patterns supported                             │
│    - Persistent checkpointer (Postgres/SQLite)                              │
│    - Multiple sequential approvals                                          │
│    - State resumability across restarts                                     │
│    - HITL interrupts survive process restart                                │
│    - Streaming support                                                      │
│                                                                             │
│ ⚠️ MUST IMPLEMENT YOURSELF (HITL):                                          │
│    - Audit logging (who approved, when)                                     │
│    - Timeout handling (stale approval requests)                             │
│    - Notification system (email/Slack/webhook)                              │
│    - Authorization (who can approve what)                                   │
│    - User context tracking                                                  │
│                                                                             │
│ ⚠️ MUST IMPLEMENT YOURSELF (DURABLE EXECUTION):                             │
│    - Checkpoint cleanup (old checkpoints accumulate)                        │
│    - Thread listing API (query DB directly)                                 │
│    - Unique thread_id generation (avoid race conditions)                    │
│    - State schema versioning (for migrations)                               │
│    - Checkpoint size monitoring                                             │
│                                                                             │
│ 📝 VERDICT:                                                                 │
│    LangGraph provides solid core primitives for:                            │
│    - Graph execution with state                                             │
│    - Human-in-the-loop interrupts                                           │
│    - Durable execution with checkpoints                                     │
│                                                                             │
│    But for production, you need to build:                                   │
│    - Approval management layer (UI, API, notifications)                     │
│    - Audit/compliance layer                                                 │
│    - Checkpoint cleanup jobs                                                │
│    - Thread management system                                               │
│    - Monitoring and alerting                                                │
│                                                                             │
│    Estimate: 3-5x effort for surrounding infrastructure                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""

if __name__ == "__main__":
    print(SUMMARY)
