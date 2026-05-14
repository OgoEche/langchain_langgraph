from State import AgentTypeOutput, AgentType, AgentState
from langgraph.types import Command
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv(override=True)


llm_model = ChatOpenAI(model="gpt-5-mini", use_responses_api=True)

# Structured LLM for routing
llm_router = llm_model.with_structured_output(AgentTypeOutput)

# -----------------------------------------------------------------------------
# Router Agent System Prompt Constant
# -----------------------------------------------------------------------------
ROUTER_SYSTEM_PROMPT = (
    """You are a router. Given the following user message, 
    decide if it is a travel information question 
    (about destinations, attractions, or general travel info) """
    """or an accommodation booking question (about hotels, 
    BnBs, room availability, or prices).\n"""
    """If it is a travel information question, 
    respond with 'travel_info_agent'.\n"""
    """If it is an accommodation booking question, 
    respond with 'accommodation_booking_agent'."""
)



# -----------------------------------------------------------------------------
# Router Agent Node for LangGraph (with structured output)
# -----------------------------------------------------------------------------
def router_agent_node(state: AgentState) -> Command[AgentType]:
    """Router node: decides which agent should handle the user query."""
    messages = state["messages"] #A
    last_msg = messages[-1] if messages else None #B
    if isinstance(last_msg, HumanMessage): #C
        user_input = last_msg.content #D
        router_messages = [ #E
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=user_input)
        ]
        router_response = llm_router.invoke(router_messages) #F
        agent_name = router_response.agent.value #G
        return Command(update=state, goto=agent_name) #H
    
    return Command(update=state, goto=AgentType.travel_info_agent) #I

#A Get the messages from the state
#B Get the last message from the messages list
#C Check if the last message is a HumanMessage
#D Get the content of the last message
#E Create the router messages, including the system prompt and the user input
#F Invoke the router model, which returns the relevant agent name
#G Get the agent name from the router response
#H Return the command to update the state and go to the agent
#I If the last message is not a HumanMessage, return the command to update the state and go to the travel_info_agent (default agent)

