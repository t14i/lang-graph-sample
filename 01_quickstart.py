"""
LangGraph Quick Start - Minimal configuration
State definition -> Node addition -> Edge connection -> compile & invoke
"""

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic


# 1. State definition
class State(TypedDict):
    messages: Annotated[list, add_messages]


# 2. LLM initialization
llm = ChatAnthropic(model="claude-sonnet-4-20250514")


# 3. Node definition
def chatbot(state: State) -> State:
    """Simple chatbot node"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# 4. Graph construction
graph_builder = StateGraph(State)

# Add node
graph_builder.add_node("chatbot", chatbot)

# Connect edges
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile
graph = graph_builder.compile()


# 5. Execute
if __name__ == "__main__":
    result = graph.invoke({"messages": [("user", "What is LangGraph? One sentence.")]})
    print("=== Result ===")
    print(result["messages"][-1].content)
