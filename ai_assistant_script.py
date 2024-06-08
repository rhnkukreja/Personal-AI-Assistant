import streamlit as st
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.sqlite import SqliteSaver
from prompts import prompt

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

class Agent:
    def __init__(self, model, tools, checkpointer, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}

# Set up the components
memory = SqliteSaver.from_conn_string(":memory:")
tool = TavilySearchResults(max_results=2)
model = ChatOpenAI(model="gpt-4-turbo")
agent_bot = Agent(model, [tool], checkpointer=memory, system=prompt)

# Streamlit UI
st.title("Personal AI Assistant")
st.write("Powered by langgraph from langchain and Tavily search API")
st.write("Ask a question and get real-time answers and recommendations!")

# Input field for the query
query = st.text_input("Enter your query:")

if st.button("Search"):
    # If the search button is clicked
    if query:
        messages = [HumanMessage(content=query)]
        thread = {"configurable": {"thread_id": "2"}}
        result = agent_bot.graph.invoke({"messages": messages}, thread)

        # Display the result in a box
        st.text_area("Answer", value=result['messages'][-1].content, height=600)
    else:
        st.warning("Please enter a query.")

