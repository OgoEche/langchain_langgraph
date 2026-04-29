from dotenv import load_dotenv
import os
import operator
from typing import Annotated, Sequence, TypedDict
import json
import asyncio

from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition

from langsmith import traceable

import nest_asyncio
nest_asyncio.apply()
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

# async def build_travel_info_vector_store(destinations: Sequence[str]) -> Chroma:
#     """Download Wikipedia pages and create a Chroma vector store."""

#     headers = {"User-Agent": "TravelBot/1.0 (educational; contact test@example.com)"}
#     converter = html2text.HTML2Text()
#     converter.ignore_links = True

#     documents = []
#     print("Downloading destination pages ...")

#     async with aiohttp.ClientSession(headers=headers) as session:
#         for slug in destinations:
#             params = {
#                 "action": "parse",
#                 "page": slug,
#                 "prop": "text",
#                 "format": "json"
#             }
#             async with session.get(WIKIPEDIA_API, params=params) as response:
#                 data = await response.json()
#                 if "error" in data:
#                     print(f"Skipping {slug}: {data['error']}")
#                     continue
#                 html = data["parse"]["text"]["*"]
#                 plain_text = converter.handle(html)
#                 documents.append(Document(
#                     page_content=plain_text,
#                     metadata={"source": f"https://en.wikipedia.org/wiki/{slug}"}
#                 ))
                

#     # Split the documents into chunks
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
#     chunks = sum([splitter.split_documents([d]) for d in documents], [])

#     # Create embeddings for the chunks
#     print(f"Embedding {len(chunks)} chunks ...")
#     embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1024)
#     vector_store = Chroma.from_documents(chunks, embeddings)

#     return vector_store

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
#@traceable(run_type="tool")
@tool("search_travel_info", return_direct=True)
def search_travel_info(query: str) -> str:
    """semantic search for travel information related to the query in the embedded WikiVoyage content for 
    information about destinations in England."""
    docs = retriever.invoke(query)
    top = docs[:4] if isinstance(docs, list) else docs
    
    return "\n---\n".join(d.page_content for d in top)


# configure LLM with tool awareness
TOOLS = [search_travel_info]

llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)

# Initialize the dependences for the Langgraph graph
# Define the agent state
# The agent state only contains LLM messages, which are appended to the list of messages
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# A CustomToolNode node class that receives tool call requests from the LLM, 
# executes each tool with its arguments, and returns the results.
# It is instantiated once and wired into the LangGraph graph as the dedicated tool execution step.
@traceable(run_type="tool")
class ToolsExecutionNode: 
    """Execute tools requested by the LLM in the last AIMessage."""

    def __init__(self, tools: Sequence): 
        self._tools_by_name = {t.name: t for t in tools}

    def __call__(self, state: dict):
        messages: Sequence[BaseMessage] = state.get("messages", [])  

        last_msg = messages[-1] 
        tool_messages: list[ToolMessage] = [] 
        tool_calls = getattr(last_msg, 
            "tool_calls", []) 
        
        for tool_call in tool_calls: 
            tool_name = tool_call["name"] 
            tool_args = tool_call["args"] 
            tool = self._tools_by_name[tool_name] 
            result = tool.invoke(tool_args) 
            tool_messages.append(
                ToolMessage(
                    content=json.dumps(result),
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": tool_messages} 

tools_execution_node = ToolsExecutionNode(TOOLS)

# LLM node
# A node function that passes the current message history to the LLM and returns its response.
# The LLM decides at this point whether to call a tool or produce a final answer.
def llm_node(state: AgentState):  
    """LLM node that decides whether to call the search tool."""
    current_messages = state["messages"] 
    respose_message = llm_with_tools.invoke(current_messages)

    return {"messages": [respose_message]}


# Build the LangGraph graph (llm_node + CustomToolNode)
# Wires an LLM node and tools node into a graph with conditional edges routing 
# between tool execution and final answer.
# Sets the LLM as entry point, adds a tools→LLM loop, and compiles.

builder = StateGraph(AgentState)
builder.add_node("llm_node", llm_node)
builder.add_node("tools", tools_execution_node)
builder.add_conditional_edges("llm_node", tools_condition)
builder.add_edge("tools", "llm_node") 

builder.set_entry_point("llm_node") 
travel_info_agent = builder.compile() 


#draw the graph
travel_info_agent.get_graph().draw_mermaid_png(output_file_path="travel_info_agent.png")

def main_chat_loop():
    print("Travel Assistant (type 'exit' to quit)")
    while True:
        user_input = input("You: ").strip() 
        if user_input.lower() in  {"exit", "quit"}: 
                break
        state = {"messages":  [HumanMessage(content=user_input)]} 
        result = travel_info_agent.invoke(state)
        response_msg = result["messages"][-1] 
        print(f"Assistant: {response_msg.content}\n") 


if __name__ == "__main__":
    main_chat_loop()
