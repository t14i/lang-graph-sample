"""
LangGraph HITL - Reject and Edit cases
Testing approve / reject / edit scenarios
"""

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage


# 1. State definition
class State(TypedDict):
    messages: Annotated[list, add_messages]


# 2. Tool definition
@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email. Requires human approval."""
    return f"Email sent to {to} with subject '{subject}'"


tools = [send_email]


# 3. LLM with tools
llm = ChatAnthropic(model="claude-sonnet-4-20250514")
llm_with_tools = llm.bind_tools(tools)


# 4. Node definitions
def agent(state: State) -> State:
    """Agent node"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def human_approval(state: State) -> Command:
    """Human approval node with approve/reject/edit support"""
    last_message = state["messages"][-1]

    if not last_message.tool_calls:
        return Command(goto="agent")

    tool_call = last_message.tool_calls[0]

    # Interrupt and wait for human decision
    decision = interrupt({
        "action": "approve_tool_call",
        "tool_name": tool_call["name"],
        "tool_args": tool_call["args"],
        "options": ["approve", "reject", "edit"],
    })

    action = decision.get("action", "reject")

    if action == "approve":
        # Continue to tool execution
        return Command(goto="tools")

    elif action == "reject":
        # Skip tool, add rejection message
        rejection_msg = ToolMessage(
            content=f"Tool call rejected by human: {decision.get('reason', 'No reason provided')}",
            tool_call_id=tool_call["id"],
        )
        return Command(goto="agent", update={"messages": [rejection_msg]})

    elif action == "edit":
        # Modify tool args and continue
        edited_args = decision.get("edited_args", tool_call["args"])
        # Update the tool call with edited args
        last_message.tool_calls[0]["args"] = edited_args
        return Command(goto="tools", update={"messages": [last_message]})

    return Command(goto="agent")


def should_continue(state: State) -> str:
    """Conditional edge"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "human_approval"
    return END


# 5. Graph construction
graph_builder = StateGraph(State)

graph_builder.add_node("agent", agent)
graph_builder.add_node("human_approval", human_approval)
graph_builder.add_node("tools", ToolNode(tools))

graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", should_continue, ["human_approval", END])
graph_builder.add_edge("tools", "agent")

checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)


def run_test(test_name: str, thread_id: str, initial_message: str, resume_decision: dict):
    """Helper to run test scenario"""
    print(f"\n{'='*50}")
    print(f"=== {test_name} ===")
    print(f"{'='*50}")

    config = {"configurable": {"thread_id": thread_id}}

    # Initial request
    result = graph.invoke({"messages": [("user", initial_message)]}, config=config)

    state = graph.get_state(config)
    if state.next:
        print(f"Interrupted at: {state.next}")
        print(f"Tool call: {state.tasks[0].interrupts[0].value}")
        print(f"\nResuming with: {resume_decision}")

        result = graph.invoke(Command(resume=resume_decision), config=config)
        print(f"\nFinal response: {result['messages'][-1].content}")
    else:
        print(f"No interrupt, direct response: {result['messages'][-1].content}")


if __name__ == "__main__":
    # Test 1: Approve
    run_test(
        "Test 1: APPROVE",
        "thread-approve",
        "Send an email to alice@example.com with subject 'Meeting' and body 'Tomorrow at 3pm'",
        {"action": "approve"}
    )

    # Test 2: Reject
    run_test(
        "Test 2: REJECT",
        "thread-reject",
        "Send an email to bob@example.com with subject 'Spam' and body 'Buy now!'",
        {"action": "reject", "reason": "This looks like spam"}
    )

    # Test 3: Edit
    run_test(
        "Test 3: EDIT",
        "thread-edit",
        "Send an email to wrong@example.com with subject 'Hello' and body 'Hi there'",
        {
            "action": "edit",
            "edited_args": {
                "to": "correct@example.com",
                "subject": "Hello",
                "body": "Hi there"
            }
        }
    )
