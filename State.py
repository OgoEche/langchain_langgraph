from typing import Annotated, Sequence, TypedDict, Literal, Optional, List, Dict
import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from enum import Enum
from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# AgentState: it only contains LLM messages
# -----------------------------------------------------------------------------
class AgentState(TypedDict): #A
    messages: Annotated[Sequence[BaseMessage], operator.add]
  
#Define the agent state
# this is a special type of state that contains the remaining steps of the agent

# -----------------------------------------------------------------------------
# AgentType Enum and Structured Output Model
# -----------------------------------------------------------------------------
class AgentType(str, Enum):
    travel_info_agent = "travel_info_agent"
    accommodation_booking_agent = "accommodation_booking_agent"

class AgentTypeOutput(BaseModel): 
    agent: AgentType = Field(..., 
    description="Which agent should handle the query?")