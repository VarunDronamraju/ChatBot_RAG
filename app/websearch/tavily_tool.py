import os
import requests

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

def search_web(query, return_source=False):
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "advanced",
        "max_results": 3
    }
    try:
        res = requests.post(url, json=payload)
        res.raise_for_status()
        data = res.json()
        if not data["results"]:
            return ("", "") if return_source else ""
        combined_content = "\n\n".join([r["content"] for r in data["results"]])
        first_url = data["results"][0]["url"]
        return (combined_content, first_url) if return_source else combined_content
    except Exception as e:
        print("❌ Tavily API error:", e)
        return ("", "") if return_source else ""
