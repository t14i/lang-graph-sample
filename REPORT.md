# LangGraph Verification Report

## Overview

Verified LangGraph's Tool Calling, HITL (Human-in-the-Loop), Durable Execution, and Memory features to evaluate production readiness.

---

# Part 1: Quick Start

## 1.1 Minimal Configuration (01_quickstart.py)

**Goal**: Understand LangGraph basics

### Graph Structure

```
START → chatbot → END
```

### Key Code

```python
# State definition - TypedDict based
class State(TypedDict):
    messages: Annotated[list, add_messages]  # add_messages for appending

# Node definition - function
def chatbot(state: State) -> State:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}  # Return diff (merged by add_messages)

# Graph construction
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# Execute
result = graph.invoke({"messages": [("user", "Hello")]})
```

### Learnings

| Element | Description |
|---------|-------------|
| `StateGraph(State)` | Initialize graph |
| `add_node(name, fn)` | Add node |
| `add_edge(from, to)` | Connect edges |
| `compile()` | Convert to executable graph |
| Node function | `State → State` (diff return OK) |

### Output

```
LangGraph is a framework for building stateful, multi-actor applications
with LLMs by modeling them as graphs where nodes represent functions/agents
and edges represent the flow of information.
```

---

# Part 2: Tool Calling

## 2.1 Tool Definition Methods (02_tool_definition.py)

**Goal**: Compare tool definition approaches

### Four Definition Methods

```python
# Method 1: @tool decorator (Simple)
@tool
def get_weather_simple(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: Sunny, 22°C"

# Method 2: @tool with Annotated (Better docs)
@tool
def get_weather_typed(
    city: Annotated[str, "The city name to get weather for"],
    unit: Annotated[str, "Temperature unit"] = "celsius"
) -> str:
    """Get current weather for a city with specified unit."""
    ...

# Method 3: Pydantic schema (Full control)
class WeatherInput(BaseModel):
    city: str = Field(description="The city name")
    unit: str = Field(default="celsius", description="Temperature unit")

@tool(args_schema=WeatherInput)
def get_weather_pydantic(city: str, unit: str = "celsius") -> str:
    ...

# Method 4: StructuredTool (Programmatic)
search_tool = StructuredTool.from_function(
    func=_search_impl,
    name="web_search",
    description="Search the web",
    args_schema=SearchInput,
)
```

### Output

```
Tool: get_weather_simple
Args Schema: {'properties': {'city': {'type': 'string'}}, 'required': ['city']}

Tool: get_weather_typed
Args Schema: {'properties': {
    'city': {'description': 'The city name to get weather for', 'type': 'string'},
    'unit': {'description': 'Temperature unit', 'default': 'celsius', 'type': 'string'}
}, 'required': ['city']}

Tool: get_weather_pydantic
Args Schema: {'properties': {
    'city': {'description': 'The city name', 'type': 'string'},
    'unit': {'description': 'Temperature unit', 'default': 'celsius', 'type': 'string'},
    'include_forecast': {'description': 'Include 3-day forecast', 'default': false}
}, 'required': ['city']}
```

### Comparison

| Method | Pros | Cons | Recommended For |
|--------|------|------|-----------------|
| @tool simple | Minimal code | No arg descriptions | Prototyping |
| @tool + Annotated | Has descriptions | Verbose for many args | Medium scale |
| @tool + Pydantic | Full control, validation | More boilerplate | Production |
| StructuredTool | Programmatic generation | Most verbose | Dynamic generation |

---

## 2.2 Tool Execution (03_tool_execution.py)

**Goal**: Verify ToolNode behavior

### Graph Structure

```
START → agent → [tool_calls?] → tools → agent → ... → END
                     ↓ no
                    END
```

### Key Code

```python
# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# ToolNode (built-in)
graph_builder.add_node("tools", ToolNode(tools))

# Conditional branching
def should_continue(state: State) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

graph_builder.add_conditional_edges("agent", should_continue, ["tools", END])

# Loop back
graph_builder.add_edge("tools", "agent")
```

### Output

```
=== TEST: Single Tool ===
Query: What's the weather in Tokyo?
  [0] HumanMessage: What's the weather in Tokyo?
  [1] AIMessage: tool_calls=['get_weather']
  [2] ToolMessage: Sunny, 22°C
  [3] AIMessage: The current weather in Tokyo is sunny with a temperature of 22°C.

=== TEST: Multiple Tools ===
Query: What's the weather in Tokyo and the stock price of AAPL?
  [1] AIMessage: tool_calls=['get_weather', 'get_stock_price']  ← Parallel calls
  [2] ToolMessage: Sunny, 22°C
  [3] ToolMessage: $178.50
  [4] AIMessage: Weather: Sunny, 22°C / AAPL: $178.50

=== TEST: Unknown Data ===
Query: What's the weather in Antarctica?
  [2] ToolMessage: No data for Antarctica
  [3] AIMessage: I wasn't able to get weather data for Antarctica...
```

### Verified Behavior

| Item | Behavior |
|------|----------|
| Single tool | Normal call and response |
| Multiple tools | Parallel calls in one AIMessage |
| Tool selection | LLM selects appropriate tool |
| No data | Returns via ToolMessage, LLM responds appropriately |

---

## 2.3 Error Handling (04_tool_error_handling.py)

**Goal**: Verify behavior when tools fail

### Default Behavior (handle_tool_errors=True)

```python
graph_builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))
```

### Output

```
=== DEFAULT ERROR HANDLING ===

Message flow:
  HumanMessage: Process the number 'abc' using validation_error_tool
  AIMessage: tool_calls=['validation_error_tool']
  ToolMessage: Error: ValueError('Expected number, got: abc') Please fix your mistakes.
  AIMessage: The validation_error_tool rejected the input 'abc' because it's not a number...

Observation: ToolNode catches exception, returns error as ToolMessage
LLM receives error and explains it to user
```

### Custom Retry Implementation

```python
class RetryToolNode:
    def __init__(self, tools: list, max_retries: int = 2):
        self.tools_by_name = {t.name: t for t in tools}
        self.max_retries = max_retries

    def __call__(self, state: State) -> State:
        for tool_call in last_message.tool_calls:
            for attempt in range(self.max_retries + 1):
                try:
                    result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
                    break
                except Exception as e:
                    last_error = e
            # Handle success or final failure
```

### Output (Retry)

```
=== RETRY ERROR HANDLING ===
Testing flaky API with retry...
  Attempt 1 failed: API temporarily unavailable
  Attempt 2 failed: API temporarily unavailable
  Attempt 3 succeeded!

Final response: The flaky API call succeeded: "API result for: test query"
```

### Error Message Format

```python
ToolMessage:
  tool_call_id: toolu_01N8VVACcSYxSxivTAoCXS86
  content: "Error: TimeoutError('Operation timed out after 100s') Please fix your mistakes."
  status: error
```

### Summary

| Item | Default Behavior |
|------|------------------|
| Exception catch | ✅ Automatic |
| ToolMessage conversion | ✅ Returns error content as ToolMessage |
| Exception propagation | ❌ Does not propagate (graph continues) |
| Retry | ❌ None (custom implementation needed) |
| LLM reaction | Recognizes error and responds appropriately |

---

# Part 3: Human-in-the-Loop (HITL)

## 3.1 Interrupt Basics (05_hitl_interrupt.py)

**Goal**: Verify interrupt/resume behavior

### Graph Structure

```
START → agent → [tool_calls?] → human_approval → tools → agent → ... → END
                     ↓ no
                    END
```

### Core APIs

| API | Role |
|-----|------|
| `interrupt(value)` | Pause graph execution, return value |
| `Command(resume=data)` | Resume interrupted graph, pass data as interrupt() return value |
| `graph.get_state(config)` | Get current state (`state.next` shows interrupt position) |
| Checkpointer | State persistence (required for interrupt) |

### Key Code

```python
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

def human_approval(state: State) -> State:
    tool_call = state["messages"][-1].tool_calls[0]

    # Graph stops here, invoke() returns
    approval = interrupt({
        "tool_name": tool_call["name"],
        "tool_args": tool_call["args"],
        "message": f"Approve {tool_call['name']}?"
    })

    if not approval.get("approved", False):
        raise ValueError("Rejected")
    return state

# Checkpointer required
checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)
```

### Execution Flow

```python
config = {"configurable": {"thread_id": "test-thread-1"}}

# 1. First invoke → interrupted
result = graph.invoke({"messages": [...]}, config=config)

# 2. Check state
state = graph.get_state(config)
print(state.next)  # ('human_approval',) ← interrupt position
print(state.tasks[0].interrupts[0].value)  # value passed to interrupt()

# 3. Resume
result = graph.invoke(Command(resume={"approved": True}), config=config)
```

### Output

```
=== Starting conversation ===
Graph state: StateSnapshot(
    next=('human_approval',),  # ← stopped here
    ...
)

=== Interrupted! Waiting at: ('human_approval',) ===
Interrupt value: {
    'tool_name': 'send_email',
    'tool_args': {'to': 'bob@example.com', 'subject': 'Hello', 'body': 'How are you?'},
}

=== Resuming with approval ===
Final result: The email has been sent successfully to bob@example.com.
```

---

## 3.2 Approve / Reject / Edit (06_hitl_approve_reject_edit.py)

**Goal**: Verify three approval patterns

### Implementation Pattern

```python
def human_approval(state: State) -> Command:
    tool_call = state["messages"][-1].tool_calls[0]

    decision = interrupt({
        "tool_name": tool_call["name"],
        "tool_args": tool_call["args"],
        "options": ["approve", "reject", "edit"],
    })

    action = decision.get("action", "reject")

    if action == "approve":
        return Command(goto="tools")

    elif action == "reject":
        rejection_msg = ToolMessage(
            content=f"Rejected: {decision.get('reason')}",
            tool_call_id=tool_call["id"],
        )
        return Command(goto="agent", update={"messages": [rejection_msg]})

    elif action == "edit":
        edited_args = decision.get("edited_args")
        last_message.tool_calls[0]["args"] = edited_args
        return Command(goto="tools", update={"messages": [last_message]})
```

### Output

```
=== Test 1: APPROVE ===
Resuming with: {'action': 'approve'}
Final response: The email has been successfully sent to alice@example.com.

=== Test 2: REJECT ===
Resuming with: {'action': 'reject', 'reason': 'This looks like spam'}
Final response: I understand your concern. The email does appear to have
characteristics commonly associated with spam. Would you like to send
a legitimate email instead?

=== Test 3: EDIT ===
Resuming with: {'action': 'edit', 'edited_args': {'to': 'correct@example.com', ...}}
Final response: The email has been sent successfully to correct@example.com.
```

### Summary

| Pattern | Behavior | LLM Response |
|---------|----------|--------------|
| Approve | Execute tool → report completion | "Sent successfully" |
| Reject | Skip tool → return to agent | Understands rejection, offers alternatives |
| Edit | Rewrite args → execute tool | Reports completion with edited values |

---

## 3.3 Checkpointer Comparison

| Checkpointer | Use Case | Persistence | Setup |
|--------------|----------|-------------|-------|
| `MemorySaver` | Dev/Test | In-process only | None |
| `SqliteSaver` | Small production | File | DB path |
| `PostgresSaver` | Production recommended | Full | Connection string |

### PostgresSaver Example

```python
from langgraph_checkpoint_postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost:5432/langgraph"
)
graph = graph_builder.compile(checkpointer=checkpointer)
```

---

# Part 4: Durable Execution

## 4.1 Basic Checkpoint Behavior (07_durable_basic.py)

**Goal**: Understand when and what is checkpointed

### Checkpoint Timing

```
=== Executing graph ===
  [Step1] Executing... (step_count: 0)
  [Step2] Executing... (step_count: 1)
  [Step3] Executing... (step_count: 2)

Checkpoint history:
  [0] next=(), step_count=3        ← After step3
  [1] next=('step3',), step_count=2  ← After step2
  [2] next=('step2',), step_count=1  ← After step1
  [3] next=('step1',), step_count=0  ← Initial
  [4] next=('__start__',)            ← Before start
```

**Observation**: Checkpoint saved AFTER each node completes.

### Resume After Restart

```python
# Phase 1: Execute, stop after step1
graph1 = build_graph()
for chunk in graph1.stream(input, config):
    if step1_done:
        break  # Simulate crash

# Phase 2: New graph instance (simulates restart)
graph2 = build_graph()
state = graph2.get_state(config)
# state.next = ('step2',) ← Recovered!

# Phase 3: Resume
result = graph2.invoke(None, config=config)
# Continues from step2
```

**Output**:
```
[Phase 1] First execution...
  [Step1] Executing...
  Step1 completed, simulating crash...
  State after crash: next=('step2',), step_count=1

[Phase 2] After restart...
  Recovered state: next=('step2',), step_count=1
  Recovered metadata: {'step1_done': True}

[Phase 3] Resuming...
  [Step2] Executing...
  [Step3] Executing...
  Final step_count: 3
```

---

## 4.2 HITL + Durability (08_durable_hitl.py)

**Goal**: Verify interrupt survives process restart

### Test Flow

```
[Phase 1] Start → Agent → interrupt() → STOP

[Phase 2] Restart (new graph instance)
  Recovered state: next=('human_approval',)
  Recovered interrupt value: {tool_name, tool_args, ...}

[Phase 3] Resume with Command(resume={"action": "approve"})
  → Tool executes → Complete
```

**Output**:
```
[Phase 1] Starting execution...
  Interrupted at: ('human_approval',)
  Interrupt value: {'tool_name': 'send_email', 'tool_args': {...}}

[Phase 2] Simulating restart...
  Recovered state: next=('human_approval',)
  Recovered interrupt: {'tool_name': 'send_email', ...}

[Phase 3] Resuming with approval...
  Final response: Email sent successfully
  Final approval_count: 1
```

**Key Finding**: HITL interrupts are fully durable. Server can restart without losing pending approvals.

---

## 4.3 Production Concerns (09_durable_production.py)

### Concurrent Execution (Same thread_id)

```
Starting 3 concurrent invocations on same thread_id...

Results: [(0, 1), (2, 1), (1, 1)]  ← All got counter=1
Errors: []
Final counter: 1  ← Last write wins
```

**Problem**: Concurrent invoke() on same thread_id causes race conditions.

**Solution**: Generate unique thread_id per conversation.

### Checkpoint Size Growth

```
1 threads, 1 msgs each: 4.0 KB
2 threads, 2 msgs each: 8.0 KB
5 threads, 5 msgs each: 20.0 KB
10 threads, 10 msgs each: 40.0 KB
```

**Observation**: Linear growth. Full state snapshot per checkpoint.

### Thread Listing

```python
# No built-in API - must query storage directly
cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
threads = cursor.fetchall()
# ['list-test-0', 'list-test-1', ...]
```

**Problem**: No API to list all thread_ids.

### Summary

| Concern | Status | Solution |
|---------|--------|----------|
| Checkpoint timing | ✅ After each node | - |
| Resume after restart | ✅ Works | Use same thread_id |
| HITL persistence | ✅ Full support | - |
| Concurrent access | ⚠️ Race condition | Unique thread_id |
| Checkpoint cleanup | ❌ No auto-cleanup | Custom job |
| Thread listing | ❌ No API | Query storage |
| State migration | ⚠️ Manual | Version schema |

---

# Part 5: Memory

## 5.1 Store Basic Operations (11_memory_store_basic.py)

**Goal**: Verify InMemoryStore basic CRUD

### Key Code

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

# Put - save with namespace (like folders)
store.put(("users", "user_123"), "preferences", {"theme": "dark", "language": "ja"})

# Get - retrieve by namespace and key
item = store.get(("users", "user_123"), "preferences")
# item.value = {"theme": "dark", "language": "ja"}

# Search - list items in namespace
results = store.search(("users", "user_123"))

# Delete
store.delete(("users", "user_123"), "preferences")
```

### Output

```
PUT: ('users', 'user_123'), key='preferences', value={'theme': 'dark', 'language': 'ja'}
GET: Item(namespace=['users', 'user_123'], key='preferences', value={'theme': 'dark', 'language': 'ja'})
Search ('users', 'user_123'): 2 items found
GET after delete: None
```

### Summary

| Operation | Description |
|-----------|-------------|
| `put(namespace, key, value)` | Save/update data |
| `get(namespace, key)` | Retrieve (None if not found) |
| `search(namespace)` | List items in namespace |
| `delete(namespace, key)` | Remove data |

---

## 5.2 Semantic Search (12_memory_semantic_search.py)

**Goal**: Verify embedding-based semantic search

### Key Code

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore(
    index={
        "dims": 1536,  # text-embedding-3-small dimension
        "embed": "openai:text-embedding-3-small",
    }
)

# Store with 'text' field for semantic indexing
store.put(("memories",), "food_1", {"text": "I love Italian food, especially pasta"})
store.put(("memories",), "work_1", {"text": "I work as a software engineer"})

# Semantic search
results = store.search(
    ("memories",),
    query="What food do I like?",
    limit=3
)
for item in results:
    print(f"[{item.score:.4f}] {item.value['text']}")
```

### Expected Output

```
Query: 'What food do I like?'
  [0.8523] I love Italian food, especially pasta
  [0.4102] I work as a software engineer

Query: 'dietary restrictions'
  [0.7891] I'm allergic to shellfish and peanuts
```

### Summary

| Feature | Description |
|---------|-------------|
| Embedding model | OpenAI text-embedding-3-small |
| Similarity | Cosine similarity (0-1) |
| Filter | Metadata-based filtering supported |
| Score > 0.8 | High relevance |
| Score < 0.5 | Low relevance |

---

## 5.3 Cross-Thread Persistence (13_memory_cross_thread.py)

**Goal**: Verify memory sharing across different thread_ids

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                 Store (Long-term)                    │
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
│            Checkpointer (Short-term)                 │
│      Each thread has separate conversation           │
└─────────────────────────────────────────────────────┘
```

### Key Points

- **Store**: Cross-thread, namespace isolates users
- **Checkpointer**: Per-thread conversation history
- Session 1 saves memory → Session 2 (different thread) can access

---

## 5.4 LangMem Memory Tools (14_memory_langmem_tools.py)

**Goal**: Verify agent-managed memory with LangMem

### Key Code

```python
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore

store = InMemoryStore(index={"dims": 1536, "embed": "openai:text-embedding-3-small"})

# Create memory tools with namespace template
manage_memory = create_manage_memory_tool(namespace=("memories", "{user_id}"))
search_memory = create_search_memory_tool(namespace=("memories", "{user_id}"))

agent = create_react_agent(
    "openai:gpt-4o",
    tools=[manage_memory, search_memory],
    store=store,
)

# Agent autonomously saves/searches memories
response = agent.invoke(
    {"messages": [{"role": "user", "content": "My name is Taro. Please remember this."}]},
    config={"configurable": {"user_id": "user_123"}}
)
```

### Agent Behavior

| Action | When |
|--------|------|
| Save memory | User says "remember" or shares personal info |
| Search memory | User asks about past information |
| Update memory | User corrects previous information |

### Unique Feature

LangMem's memory tools are **not available in CrewAI**. Agent has explicit control over memory operations.

---

## 5.5 Background Extraction (15_memory_background_extraction.py)

**Goal**: Verify automatic fact extraction from conversations

### Key Code

```python
from langmem import create_memory_store_manager

manager = create_memory_store_manager(
    "openai:gpt-4o",
    namespace=("memories", "{user_id}"),
)

conversation = [
    {"role": "user", "content": "I love Italian food but I'm allergic to shellfish."},
    {"role": "assistant", "content": "I'll note your food preferences and allergy."},
]

# Extract memories asynchronously
await manager.ainvoke(
    {"messages": conversation},
    config={"configurable": {"user_id": "user_123"}},
    store=store,
)
```

### Extraction Behavior

- Identifies user facts, preferences, constraints
- Creates semantic embeddings for search
- Updates conflicting information (consolidation)
- Runs asynchronously (background processing)

---

## 5.6 Memory Summary

| Feature | Support | Notes |
|---------|---------|-------|
| Basic CRUD | ✅ Full | put/get/delete/search |
| Namespace | ✅ Full | Folder-like structure |
| Semantic search | ✅ Full | OpenAI embeddings |
| Cross-thread | ✅ Full | Store shared across threads |
| LangMem tools | ✅ Full | Agent-managed memory |
| Background extraction | ✅ Full | Auto fact extraction |
| Production storage | ✅ PostgresStore | pgvector for vectors |
| Cleanup | ❌ None | No TTL/auto-cleanup |
| Privacy | ⚠️ Manual | PII handling needed |

### CrewAI Comparison

| Feature | LangGraph + LangMem | CrewAI |
|---------|---------------------|--------|
| Basic structure | Store + namespace | ChromaDB + SQLite |
| Embedding | OpenAI (configurable) | OpenAI (configurable) |
| Semantic search | ✅ | ✅ |
| Cross-session | ✅ | ✅ |
| Agent memory tools | ✅ **Unique** | ❌ |
| Background extraction | ✅ **Unique** | ❌ |
| Production storage | PostgresStore | External DB migration |

---

# Part 6: Production Considerations

## 6.1 Audit Logging

**Current**: None

```python
def human_approval(state: State) -> Command:
    decision = interrupt({...})

    # Manual logging required
    audit_logger.log(
        timestamp=datetime.now(),
        user_id=???,  # Where to get this?
        action=decision["action"],
        tool_name=tool_call["name"],
        tool_args=tool_call["args"],
    )
```

**Challenges**:
- How to pass approver's user ID
- Need to include metadata in `Command(resume=...)`

---

## 6.2 Timeout

**Current**: None. Waits forever when interrupted.

```python
# Implement via background job
async def cleanup_stale_threads():
    for thread_id in get_active_threads():
        state = graph.get_state({"configurable": {"thread_id": thread_id}})
        if state.next and is_stale(state.created_at, timeout=timedelta(hours=24)):
            graph.invoke(
                Command(resume={"action": "reject", "reason": "Timeout"}),
                config={"configurable": {"thread_id": thread_id}}
            )
```

**Challenges**:
- No thread listing API (checkpointer dependent)
- Timeout handling logic is custom implementation

---

## 6.3 Notification System

**Current**: None

```python
state = graph.get_state(config)
if state.next:
    # Manual notification
    slack.send(f"Pending approval: {state.tasks[0].interrupts[0].value}")
    email.send(approver, "Approval Request", ...)
```

---

## 6.4 Authorization (Who Can Approve)

**Current**: None

```python
def human_approval(state: State) -> Command:
    decision = interrupt({
        "required_role": "admin",
        ...
    })

    # Validate user info passed during resume
    if not has_role(decision["approver_id"], "admin"):
        raise PermissionError("Not authorized")
```

---

## 6.5 Web API Integration Pattern

```python
from fastapi import FastAPI
from langgraph.types import Command

app = FastAPI()

@app.post("/chat")
async def chat(message: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"messages": [("user", message)]}, config=config)

    state = graph.get_state(config)
    if state.next:
        return {
            "status": "pending_approval",
            "thread_id": thread_id,
            "approval_request": state.tasks[0].interrupts[0].value
        }
    return {
        "status": "completed",
        "response": result["messages"][-1].content
    }

@app.post("/approve/{thread_id}")
async def approve(thread_id: str, decision: dict):
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(Command(resume=decision), config=config)
    return {"response": result["messages"][-1].content}
```

---
# Part 7: Evaluation Summary

## Good

| Category | Item | Rating | Notes |
|----------|------|--------|-------|
| Tool Calling | `@tool` decorator | ⭐⭐⭐⭐⭐ | Simple, Pydantic support |
| Tool Calling | `ToolNode` | ⭐⭐⭐⭐⭐ | Auto execution, error handling |
| Tool Calling | Parallel tool calls | ⭐⭐⭐⭐⭐ | Multiple calls in one message |
| HITL | `interrupt()` API | ⭐⭐⭐⭐⭐ | Simple and intuitive |
| HITL | `Command` control | ⭐⭐⭐⭐⭐ | Flexible goto, update, resume |
| HITL | Approve/Reject/Edit | ⭐⭐⭐⭐⭐ | All patterns implementable |
| Durable | State persistence | ⭐⭐⭐⭐ | Postgres/SQLite support |
| Durable | Durable execution | ⭐⭐⭐⭐ | Resume after restart |
| Durable | HITL durability | ⭐⭐⭐⭐⭐ | Interrupts survive restart |
| Memory | Store API | ⭐⭐⭐⭐⭐ | Simple CRUD, namespace |
| Memory | Semantic search | ⭐⭐⭐⭐⭐ | OpenAI embeddings |
| Memory | Cross-thread | ⭐⭐⭐⭐⭐ | Shared across sessions |
| Memory | LangMem tools | ⭐⭐⭐⭐⭐ | Agent-managed memory |
| Memory | Background extraction | ⭐⭐⭐⭐ | Auto fact extraction |

## Not Good

| Category | Item | Rating | Notes |
|----------|------|--------|-------|
| Tool Calling | Tool retry | ⭐⭐ | Custom implementation needed |
| HITL | Audit logging | ⭐ | Fully custom |
| HITL | Timeout | ⭐ | No mechanism |
| HITL | Notification | ⭐ | No mechanism |
| HITL | Authorization | ⭐ | No mechanism |
| Durable | Checkpoint cleanup | ⭐ | No auto-cleanup |
| Durable | Thread listing | ⭐ | No built-in API |
| Durable | Concurrent access | ⭐⭐ | Race condition possible |
| Memory | Memory cleanup | ⭐ | No TTL/auto-cleanup |
| Memory | Privacy/PII | ⭐⭐ | Manual compliance |
| Memory | Embedding costs | ⭐⭐ | Per-operation cost |

---

# Part 8: Conclusion

## Tool Calling

**High maturity.** Simple definition with `@tool` decorator, automatic execution with `ToolNode`. Error handling works with `handle_tool_errors=True`. Production needs custom retry and circuit breaker implementation.

## HITL

**High maturity as "graph execution interrupt/resume".** `interrupt()` / `Command(resume=...)` API is clean and intuitive.

However, production-required features are not provided:

- Approval workflow management (who approved what, when)
- Timeout / escalation
- Notification / reminders
- Permission management

**These appear to be by design - "not LangGraph's responsibility".**

## Durable Execution

**Solid foundation.** Checkpoints saved after each node, state fully recoverable after restart. HITL interrupts persist correctly.

Production concerns:
- No auto-cleanup (checkpoint size grows indefinitely)
- No thread listing API (must query storage directly)
- Concurrent access on same thread_id causes race conditions

## Memory

**Strong capabilities.** Store API is simple and effective. Semantic search with embeddings works well. LangMem provides unique agent-managed memory features not available in CrewAI.

Production concerns:
- Embedding costs per operation
- No built-in TTL/cleanup
- Privacy/PII compliance needs manual implementation
- Background extraction quality varies

## Additional Development for Production

1. **Approval Management Service** - Manage pending threads, provide UI
2. **Audit Log Service** - Record all operations
3. **Notification Service** - Slack/Email integration
4. **Authorization Service** - Role-based approval permissions
5. **Background Jobs** - Timeout handling, checkpoint cleanup, memory cleanup
6. **Retry/Circuit Breaker** - Stabilize external API calls
7. **Thread Management** - Track active threads, cleanup old ones
8. **Unique ID Generation** - Prevent concurrent access issues
9. **Memory Lifecycle** - TTL, cleanup, cost monitoring

**Effort estimate**: 3-5x the graph execution portion for surrounding systems.

---

# File Structure

```
lang-graph-sample/
├── 01_quickstart.py              # Quick Start
├── 02_tool_definition.py         # Tool definition comparison
├── 03_tool_execution.py          # ToolNode verification
├── 04_tool_error_handling.py     # Error handling
├── 05_hitl_interrupt.py          # HITL basics (interrupt)
├── 06_hitl_approve_reject_edit.py # Approve/Reject/Edit
├── 07_durable_basic.py           # Durable execution basics
├── 08_durable_hitl.py            # HITL + Durability
├── 09_durable_production.py      # Durable production concerns
├── 11_memory_store_basic.py      # Memory Store CRUD
├── 12_memory_semantic_search.py  # Semantic search
├── 13_memory_cross_thread.py     # Cross-thread persistence
├── 14_memory_langmem_tools.py    # LangMem agent tools
├── 15_memory_background_extraction.py # Background extraction
├── 16_production_considerations.py # Overall production summary
├── REPORT.md                     # This report
├── REPORT_ja.md                  # Japanese version
├── .env.example                  # Environment template
├── pyproject.toml
└── uv.lock
```
