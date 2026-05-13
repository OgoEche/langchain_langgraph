import random
import requests

from dotenv import load_dotenv
import os
import operator
from typing import Annotated, Literal, Optional, Sequence, TypedDict
import json
import asyncio

from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent

import nest_asyncio
nest_asyncio.apply()

from BookingService import BnBBookingService

import aiohttp
import html2text
from langchain_core.documents import Document


# load environment variables from .env file
load_dotenv(override=True)


# prepare knowledge base
UK_DESTINATIONS = [
    "Cornwall",
    "North_Cornwall",
    "Devon",
    "West_Cornwall",
    "Manchester",
    "Liverpool",
    "York",
]

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

# Preparing the travel information vector store
async def build_travel_info_vector_store(destinations: Sequence[str]) -> Chroma:
    """Download Wikipedia pages and create a Chroma vector store."""

    headers = {"User-Agent": "TravelBot/1.0 (educational; contact test@example.com)"}
    converter = html2text.HTML2Text()
    converter.ignore_links = True

    documents = []
    print("Downloading destination pages ...")

    async with aiohttp.ClientSession(headers=headers) as session:
        for slug in destinations:
            params = {
                "action": "parse",
                "page": slug,
                "prop": "text",
                "format": "json"
            }

            for attempt in range(3):  # retry up to 3 times
                async with session.get(WIKIPEDIA_API, params=params) as response:

                    # Handle rate limit
                    if response.status == 429:
                        wait = 2 ** attempt  # 1s, 2s, 4s backoff
                        print(f"Rate limited on {slug}, retrying in {wait}s ...")
                        await asyncio.sleep(wait)
                        continue

                    # Safe JSON parse
                    content_type = response.headers.get("Content-Type", "")
                    if "application/json" not in content_type:
                        print(f"Unexpected content type for {slug}: {content_type} — skipping")
                        break

                    data = await response.json()
                    if "error" in data:
                        print(f"Skipping {slug}: {data['error']}")
                        break

                    html = data["parse"]["text"]["*"]
                    plain_text = converter.handle(html)
                    documents.append(Document(
                        page_content=plain_text,
                        metadata={"source": f"https://en.wikipedia.org/wiki/{slug}"}
                    ))
                    print(f"✓ {slug}")
                    break

            await asyncio.sleep(1)  # 1 second delay between each destination

    # Split the documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    chunks = sum([splitter.split_documents([d]) for d in documents], [])

    # Create embeddings for the chunks
    print(f"Embedding {len(chunks)} chunks ...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1024)
    vector_store = Chroma.from_documents(chunks, embeddings)

    return vector_store


# Singleton pattern (build once)
_vectorstore_client: Chroma | None = None

def get_travel_info_vector_store() -> Chroma:
    """Get the travel information vector store, building it if necessary."""
    global _vectorstore_client
    if _vectorstore_client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        _vectorstore_client = asyncio.run(build_travel_info_vector_store(UK_DESTINATIONS))  
    return _vectorstore_client

vectorstore_client = get_travel_info_vector_store()
retriever = vectorstore_client.as_retriever(search_kwargs={"k": 10})


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

llm = ChatOpenAI(model="gpt-4o", temperature=0)
#llm = ChatOpenAI(model="openai/qwen3.5-4b", base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")

# Initialize the dependences for the Langgraph graph
# Define the agent state
# The agent state only contains LLM messages, which are appended to the list of messages
class AgentState(TypedDict): 
    messages: Annotated[Sequence[BaseMessage], operator.add]
  


# The ReAct agent now orchestrates the flow, tool calling, and synthesis of information. The agent's prompt instructs it to use the tools to find the information it needs, including town names for weather queries.
# The agent's state schema is defined by AgentState, which includes the conversation messages and the remaining steps for the agent to complete its task. The tools available to the agent include search_travel
travel_info_agent = create_agent(
    model=llm,
    tools=TOOLS,
    state_schema=AgentState,
    system_prompt="""You are a helpful assistant that 
    can search travel information and get the weather forecast. 
    Only use the tools to find the information you need 
    (including town names).""")


#draw the graph
travel_info_agent.get_graph().draw_mermaid_png(output_file_path="travel_info_agent.png")

def main_chat_loop():
    print("Travel Assistant (type 'exit' to quit)")
    BnBBookingService.initialize(model=llm)
    BnB_booking_agent = BnBBookingService.get_accommodation_booking_agent(AgentState)

    while True:
        user_input = input("You: ").strip() 
        if user_input.lower() in  {"exit", "quit"}: 
                break
        state = {"messages":  [HumanMessage(content=user_input)]} 
        #result = travel_info_agent.invoke(state)
        result = BnB_booking_agent.invoke(state)
        response_msg = result["messages"][-1] 
        print(f"Assistant: {response_msg.content}\n") 


if __name__ == "__main__":
    main_chat_loop()
