import os
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
client = TavilyClient(api_key=TAVILY_API_KEY)

def search_web(query: str, k: int = 3):
    response = client.search(query=query, search_depth="advanced", max_results=k)
    sources = response.get("results", [])
    combined = ""
    for result in sources:
        title = result.get("title", "")
        content = result.get("content", "")
        url = result.get("url", "")
        combined += f"{title}\n{content}\nSource: {url}\n\n"
    return combined.strip()
