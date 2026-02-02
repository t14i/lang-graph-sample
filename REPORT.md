# LangGraph Verification Report

## Overview

Verified LangGraph's Tool Calling and HITL (Human-in-the-Loop) features to evaluate production readiness.

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

# Part 4: Production Considerations

## 4.1 Audit Logging

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

## 4.2 Timeout

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

## 4.3 Notification System

**Current**: None

```python
state = graph.get_state(config)
if state.next:
    # Manual notification
    slack.send(f"Pending approval: {state.tasks[0].interrupts[0].value}")
    email.send(approver, "Approval Request", ...)
```

---

## 4.4 Authorization (Who Can Approve)

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

## 4.5 Web API Integration Pattern

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

# Part 5: Evaluation Summary

## Good

| Item | Rating | Notes |
|------|--------|-------|
| `@tool` decorator | ⭐⭐⭐⭐⭐ | Simple, Pydantic support |
| `ToolNode` | ⭐⭐⭐⭐⭐ | Auto execution, error handling |
| Parallel tool calls | ⭐⭐⭐⭐⭐ | Multiple calls in one message |
| `interrupt()` API | ⭐⭐⭐⭐⭐ | Simple and intuitive |
| `Command` control | ⭐⭐⭐⭐⭐ | Flexible goto, update, resume |
| Approve/Reject/Edit | ⭐⭐⭐⭐⭐ | All patterns implementable |
| State persistence | ⭐⭐⭐⭐ | Postgres/SQLite support |

## Not Good

| Item | Rating | Notes |
|------|--------|-------|
| Tool retry | ⭐⭐ | Custom implementation needed |
| Audit logging | ⭐ | Fully custom |
| Timeout | ⭐ | No mechanism |
| Notification | ⭐ | No mechanism |
| Authorization | ⭐ | No mechanism |

---

# Conclusion

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

## Additional Development for Production

1. **Approval Management Service** - Manage pending threads, provide UI
2. **Audit Log Service** - Record all operations
3. **Notification Service** - Slack/Email integration
4. **Authorization Service** - Role-based approval permissions
5. **Background Jobs** - Timeout handling, cleanup
6. **Retry/Circuit Breaker** - Stabilize external API calls

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
├── 07_production_considerations.py # Production considerations
├── REPORT.md                     # This report
├── REPORT_ja.md                  # Japanese version
├── .env.example                  # Environment template
├── pyproject.toml
└── uv.lock
```
