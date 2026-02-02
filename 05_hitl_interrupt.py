"""
LangGraph Human-in-the-Loop - interrupt() based implementation
interrupt() to pause graph, Command(resume=...) to resume, Checkpointer required
"""

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool


# 1. State definition
class State(TypedDict):
    messages: Annotated[list, add_messages]


# 2. Tool definition with human approval
@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email. Requires human approval."""
    # This would actually send email in production
    return f"Email sent to {to} with subject '{subject}'"


@tool
def delete_file(path: str) -> str:
    """Delete a file. Requires human approval."""
    # This would actually delete file in production
    return f"File {path} deleted"


tools = [send_email, delete_file]


# 3. LLM with tools
llm = ChatAnthropic(model="claude-sonnet-4-20250514")
llm_with_tools = llm.bind_tools(tools)


# 4. Node definitions
def agent(state: State) -> State:
    """Agent node"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def human_approval(state: State) -> State:
    """Human approval node - interrupts for approval"""
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            # Interrupt and wait for human approval
            approval = interrupt({
                "action": "approve_tool_call",
                "tool_name": tool_call["name"],
                "tool_args": tool_call["args"],
                "message": f"Approve {tool_call['name']} with args {tool_call['args']}?"
            })

            if not approval.get("approved", False):
                # If rejected, we could modify state or skip tool
                raise ValueError(f"Tool call {tool_call['name']} was rejected by human")

    return state


def should_continue(state: State) -> str:
    """Conditional edge"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "human_approval"
    return END


# 5. Graph construction
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("agent", agent)
graph_builder.add_node("human_approval", human_approval)
graph_builder.add_node("tools", ToolNode(tools))

# Add edges
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", should_continue, ["human_approval", END])
graph_builder.add_edge("human_approval", "tools")
graph_builder.add_edge("tools", "agent")

# Compile with checkpointer (required for interrupt)
checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)


# 6. Execute with interrupt handling
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "test-thread-1"}}

    print("=== Starting conversation ===")

    # Initial request
    result = graph.invoke(
        {"messages": [("user", "Send an email to bob@example.com with subject 'Hello' and body 'How are you?'")]},
        config=config
    )

    print(f"Graph state: {graph.get_state(config)}")

    # Check if interrupted
    state = graph.get_state(config)
    if state.next:
        print(f"\n=== Interrupted! Waiting at: {state.next} ===")
        print(f"Interrupt value: {state.tasks}")

        # Resume with approval
        print("\n=== Resuming with approval ===")
        result = graph.invoke(
            Command(resume={"approved": True}),
            config=config
        )
        print(f"Final result: {result['messages'][-1].content}")
    else:
        print(f"Result: {result['messages'][-1].content}")
