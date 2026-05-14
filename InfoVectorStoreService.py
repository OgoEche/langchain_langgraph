import aiohttp
import html2text
from langchain_core.documents import Document

from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from typing import Sequence
import nest_asyncio
import asyncio
import os
nest_asyncio.apply()


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