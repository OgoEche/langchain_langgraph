import random
import requests

from dotenv import load_dotenv
import operator
from typing import Annotated, Literal, Optional, Sequence, TypedDict


from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.graph import StateGraph

import nest_asyncio
nest_asyncio.apply()
from State import AgentState
from InfoVectorStoreService import retriever

# load environment variables from .env file
load_dotenv(override=True)



# define the tool for retrieving travel information and add langsmith tracing and Langgraph tool metadata
@tool("search_travel_info", return_direct=True)
def search_travel_info(query: str) -> str:
    """semantic search for travel information related to the query in the embedded WikiVoyage content for 
    information about destinations in England."""
    docs = retriever.invoke(query)
    top = docs[:4] if isinstance(docs, list) else docs
    
    return "\n---\n".join(d.page_content for d in top)

# simulate real-time weather data for any given town
class WeatherForecast(TypedDict):
    town: str
    weather:  Literal["sunny", "foggy", "rainy", "windy"]
    temperature: int

class WeatherForecastService:
    """A mock weather forecast service that returns random weather data for a given town."""
    _weather_options = ["sunny", "foggy", "rainy", "windy"]
    _temp_min = 18
    _temp_max = 31

    @classmethod
    def get_forecast(cls, town: str) -> Optional[WeatherForecast]:
        weather = random.choice(cls._weather_options)
        temperature = random.randint(cls._temp_min, cls._temp_max)
        return WeatherForecast(town=town, weather=weather, temperature=temperature)


@tool(description="Get the weather forecast, given a town name.")
def weather_forecast(town: str) -> dict:
    """Get a mock weather forecast for a given town. Returns a
     WeatherForecast object with weather and temperature."""
    forecast = WeatherForecastService.get_forecast(town)
    return {"error": f"No weather data available for '{town}'."} if forecast is None else forecast


@tool(description="Get the current weather for a specified location.")
def get_weather(location: str) -> str:
    """Get current weather for a specific city or town.
    Input must be a single place name, e.g. 'St Ives' or 'Newquay'.
    Do NOT pass descriptions or multiple locations — call once per place."""
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1}
        ).json()
        
        if "results" not in geo or not geo["results"]:
            return f"Could not find coordinates for '{location}'. Try a more specific city name."
        
        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]
        name = geo["results"][0].get("name", location)
        
        weather = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": True}
        ).json()
        
        cw = weather["current_weather"]
        return (
            f"Weather in {name}: {cw['temperature']}°C, "
            f"Wind: {cw['windspeed']} km/h, "
            f"Code: {cw['weathercode']}"
        )
    except Exception as e:
        return f"Weather lookup failed for '{location}': {e}"
   

# configure LLM with tool awareness
TOOLS = [search_travel_info,
         #weather_forecast,
         get_weather,
        ]


# The ReAct agent now orchestrates the flow, tool calling, and synthesis of information. The agent's prompt instructs it to use the tools to find the information it needs, including town names for weather queries.
# The agent's state schema is defined by AgentState, which includes the conversation messages and the remaining steps for the agent to complete its task. The tools available to the agent include search_travel
def get_travel_info_agent(agent_state, llm: ChatOpenAI) -> StateGraph:
    return create_agent(
        model=llm,
        tools=TOOLS,
        state_schema=agent_state,
        system_prompt="""You are a helpful assistant that 
        can search travel information and get the weather forecast. 
        Only use the tools to find the information you need 
        (including town names).""",
    )
