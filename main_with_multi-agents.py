import random
import requests

from dotenv import load_dotenv
import operator
from typing import Annotated, Literal, Optional, Sequence, TypedDict


from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent

import nest_asyncio
nest_asyncio.apply()

from BookingService import BnBBookingService
from TravelService  import get_travel_info_agent
from RouterService import router_agent_node
from State import AgentState

# load environment variables from .env file
load_dotenv(override=True)




llm = ChatOpenAI(model="gpt-4o", temperature=0)
#llm = ChatOpenAI(model="openai/qwen3.5-4b", base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")


# The ReAct agent now orchestrates the flow, tool calling, and synthesis of information. The agent's prompt instructs it to use the tools to find the information it needs, including town names for weather queries.
# The agent's state schema is defined by AgentState, which includes the conversation messages and the remaining steps for the agent to complete its task. The tools available to the agent include search_travel
travel_info_agent = get_travel_info_agent(AgentState, llm)

#draw the graph
travel_info_agent.get_graph().draw_mermaid_png(output_file_path="travel_info_agent.png")

# Accommodation Booking Agent
BnBBookingService.initialize(model=llm)
accommodation_booking_agent = BnBBookingService.get_accommodation_booking_agent(AgentState)


# -----------------------------------------------------------------------------
# Build the LangGraph graph with router, travel_info_agent, and accommodation_booking_agent
# -----------------------------------------------------------------------------
graph = StateGraph(AgentState) #A
graph.add_node("router_agent", router_agent_node) #B
graph.add_node("travel_info_agent", travel_info_agent) #C
graph.add_node("accommodation_booking_agent", accommodation_booking_agent) #D

graph.add_edge("travel_info_agent", END) #E
graph.add_edge("accommodation_booking_agent", END) #F

graph.set_entry_point("router_agent") #G
travel_assistant = graph.compile() #H

travel_assistant.get_graph().draw_mermaid_png(output_file_path="travel_assistant_graph.png")

#A Define the graph
#B Add the router agent node
#C Add the travel info agent node
#D Add the accommodation booking agent node
#E Add the edge from the travel info agent to the end
#F Add the edge from the accommodation booking agent to the end
#G Set the entry point to the router agent
#H Compile the graph


def main_chat_loop():
    print("Travel Assistant (type 'exit' to quit)")
    # BnBBookingService.initialize(model=llm)
    # BnB_booking_agent = BnBBookingService.get_accommodation_booking_agent(AgentState)

    while True:
        user_input = input("You: ").strip() 
        if user_input.lower() in  {"exit", "quit"}: 
                break
        state = {"messages":  [HumanMessage(content=user_input)]} 
        result = travel_assistant.invoke(state)
        response_msg = result["messages"][-1] 
        print(f"Assistant: {response_msg.content}\n") 


if __name__ == "__main__":
    main_chat_loop()
