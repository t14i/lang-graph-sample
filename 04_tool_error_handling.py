"""
Tool Calling - Part 3: Error Handling Deep Dive
What happens when tools fail? Custom error handling strategies.
"""

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage


# =============================================================================
# State with error tracking
# =============================================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]
    error_count: int


# =============================================================================
# Tools with different failure modes
# =============================================================================

@tool
def flaky_api(query: str) -> str:
    """Simulates a flaky API that sometimes fails."""
    import random
    if random.random() < 0.5:
        raise ConnectionError("API temporarily unavailable")
    return f"API result for: {query}"


@tool
def validation_error_tool(number: str) -> str:
    """Tool that validates input and may reject it."""
    if not number.isdigit():
        raise ValueError(f"Expected number, got: {number}")
    return f"Processed number: {number}"


@tool
def timeout_tool(seconds: int) -> str:
    """Tool that simulates timeout."""
    if seconds > 5:
        raise TimeoutError(f"Operation timed out after {seconds}s")
    return f"Completed in {seconds}s"


tools = [flaky_api, validation_error_tool, timeout_tool]


# =============================================================================
# Custom ToolNode with retry logic
# =============================================================================

class RetryToolNode:
    """Custom tool node with retry capability."""

    def __init__(self, tools: list, max_retries: int = 2):
        self.tool_node = ToolNode(tools)
        self.max_retries = max_retries
        self.tools_by_name = {t.name: t for t in tools}

    def __call__(self, state: State) -> State:
        last_message = state["messages"][-1]
        results = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            result = None
            last_error = None

            for attempt in range(self.max_retries + 1):
                try:
                    tool_fn = self.tools_by_name[tool_name]
                    result = tool_fn.invoke(tool_args)
                    break
                except Exception as e:
                    last_error = e
                    print(f"  Attempt {attempt + 1} failed: {e}")

            if result is not None:
                results.append(ToolMessage(content=result, tool_call_id=tool_id))
            else:
                error_msg = f"Failed after {self.max_retries + 1} attempts: {last_error}"
                results.append(ToolMessage(content=error_msg, tool_call_id=tool_id))

        return {"messages": results, "error_count": state.get("error_count", 0)}


# =============================================================================
# Graph with custom error handling
# =============================================================================

llm = ChatAnthropic(model="claude-sonnet-4-20250514")
llm_with_tools = llm.bind_tools(tools)


def agent(state: State) -> State:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response], "error_count": state.get("error_count", 0)}


def should_continue(state: State) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# Build two graphs: default and retry
def build_graph(use_retry: bool = False):
    builder = StateGraph(State)
    builder.add_node("agent", agent)

    if use_retry:
        builder.add_node("tools", RetryToolNode(tools, max_retries=2))
    else:
        # handle_tool_errors=True wraps exceptions in ToolMessage
        builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, ["tools", END])
    builder.add_edge("tools", "agent")
    return builder.compile()


# =============================================================================
# Tests
# =============================================================================

def test_default_error_handling():
    """Test how default ToolNode handles errors."""
    print("\n" + "="*60)
    print("DEFAULT ERROR HANDLING")
    print("="*60)

    graph = build_graph(use_retry=False)

    # Test validation error
    result = graph.invoke({
        "messages": [("user", "Process the number 'abc' using validation_error_tool")],
        "error_count": 0
    })

    print("\nMessage flow:")
    for msg in result["messages"]:
        print(f"  {type(msg).__name__}: {str(msg.content)[:80]}...")

    print(f"\nObservation: ToolNode catches exception, returns error as ToolMessage")
    print(f"LLM receives error and explains it to user")


def test_retry_error_handling():
    """Test custom retry logic."""
    print("\n" + "="*60)
    print("RETRY ERROR HANDLING")
    print("="*60)

    graph = build_graph(use_retry=True)

    # Test flaky API (may succeed after retry)
    print("\nTesting flaky API with retry...")
    result = graph.invoke({
        "messages": [("user", "Query the flaky_api with 'test query'")],
        "error_count": 0
    })

    print(f"\nFinal response: {result['messages'][-1].content[:200]}...")


def test_error_message_format():
    """Examine the exact format of error messages via graph."""
    print("\n" + "="*60)
    print("ERROR MESSAGE FORMAT")
    print("="*60)

    graph = build_graph(use_retry=False)

    result = graph.invoke({
        "messages": [("user", "Use timeout_tool with 100 seconds")],
        "error_count": 0
    })

    # Find ToolMessage with error
    for msg in result["messages"]:
        if type(msg).__name__ == "ToolMessage":
            print(f"\nToolMessage:")
            print(f"  tool_call_id: {msg.tool_call_id}")
            print(f"  content: {msg.content}")
            print(f"  status: {getattr(msg, 'status', 'N/A')}")


if __name__ == "__main__":
    test_default_error_handling()
    test_retry_error_handling()
    test_error_message_format()

    print("\n" + "="*60)
    print("ERROR HANDLING SUMMARY")
    print("="*60)
    print("""
Default ToolNode behavior:
- Catches all exceptions
- Wraps error in ToolMessage with error content
- Does NOT propagate exception
- LLM sees error and responds accordingly

Custom strategies:
1. RetryToolNode: Retry failed tools N times
2. FallbackToolNode: Try alternative tool on failure
3. CircuitBreakerNode: Stop calling after N failures

Production considerations:
- Log all errors with context
- Implement circuit breakers for external APIs
- Consider graceful degradation
- Track error rates for monitoring
""")
