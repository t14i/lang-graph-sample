"""
Tool Calling - Part 2: Tool Execution with ToolNode
Auto-processing, error handling, multiple tools
"""

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool


# =============================================================================
# State
# =============================================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]


# =============================================================================
# Tools - Various scenarios
# =============================================================================

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    data = {"tokyo": "Sunny, 22°C", "osaka": "Cloudy, 19°C", "london": "Rainy, 12°C"}
    return data.get(city.lower(), f"No data for {city}")


@tool
def get_stock_price(symbol: str) -> str:
    """Get current stock price for a symbol."""
    prices = {"AAPL": "$178.50", "GOOGL": "$141.20", "MSFT": "$378.90"}
    return prices.get(symbol.upper(), f"Unknown symbol: {symbol}")


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        # Safe eval for simple math
        allowed = set("0123456789+-*/.(). ")
        if all(c in allowed for c in expression):
            return str(eval(expression))
        return "Invalid expression"
    except Exception as e:
        return f"Error: {e}"


@tool
def failing_tool(input: str) -> str:
    """This tool always fails - for testing error handling."""
    raise ValueError(f"Intentional failure with input: {input}")


# =============================================================================
# Graph setup
# =============================================================================

tools = [get_weather, get_stock_price, calculate, failing_tool]
llm = ChatAnthropic(model="claude-sonnet-4-20250514")
llm_with_tools = llm.bind_tools(tools)


def agent(state: State) -> State:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if last.tool_calls else END


graph_builder = StateGraph(State)
graph_builder.add_node("agent", agent)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", should_continue, ["tools", END])
graph_builder.add_edge("tools", "agent")

graph = graph_builder.compile()


# =============================================================================
# Test scenarios
# =============================================================================

def run_test(name: str, query: str):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"Query: {query}")
    print("="*60)

    try:
        result = graph.invoke({"messages": [("user", query)]})

        # Show message flow
        for i, msg in enumerate(result["messages"]):
            msg_type = type(msg).__name__
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                print(f"  [{i}] {msg_type}: tool_calls={[tc['name'] for tc in msg.tool_calls]}")
            elif hasattr(msg, 'content'):
                content = msg.content[:100] + "..." if len(str(msg.content)) > 100 else msg.content
                print(f"  [{i}] {msg_type}: {content}")

        print(f"\nFinal: {result['messages'][-1].content}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    # Test 1: Single tool
    run_test("Single Tool", "What's the weather in Tokyo?")

    # Test 2: Multiple tools in one query
    run_test("Multiple Tools", "What's the weather in Tokyo and the stock price of AAPL?")

    # Test 3: Tool selection
    run_test("Tool Selection", "Calculate 123 * 456")

    # Test 4: Tool with unknown data
    run_test("Unknown Data", "What's the weather in Antarctica?")

    # Test 5: Error handling - tool failure
    print("\n" + "="*60)
    print("TEST: Error Handling (Tool Failure)")
    print("="*60)
    try:
        # Manually invoke failing tool via ToolNode
        from langchain_core.messages import AIMessage
        tool_node = ToolNode(tools)
        state = {
            "messages": [
                AIMessage(content="", tool_calls=[{
                    "id": "test123",
                    "name": "failing_tool",
                    "args": {"input": "test"}
                }])
            ]
        }
        result = tool_node.invoke(state)
        print(f"ToolNode result: {result}")
    except Exception as e:
        print(f"Exception: {type(e).__name__}: {e}")

    # Summary
    print("\n" + "="*60)
    print("TOOLNODE BEHAVIOR SUMMARY")
    print("="*60)
    print("""
1. Auto-execution: ToolNode automatically executes tools based on tool_calls
2. Multiple tools: Handles parallel tool calls in single message
3. Error handling: Catches exceptions, returns error as ToolMessage
4. Message flow: AIMessage(tool_calls) → ToolMessage(result) → AIMessage(response)

Key observations:
- ToolNode wraps exceptions in ToolMessage, doesn't propagate
- LLM sees error message and can respond appropriately
- No built-in retry mechanism
""")
