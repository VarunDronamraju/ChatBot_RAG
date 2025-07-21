import os
import requests

def search_web(query, include_meta=False):
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("[ERROR] Tavily API key not found in .env")
        return None

    url = "https://api.tavily.com/search"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "query": query,
        "search_depth": "basic",
        "include_answer": True,
        "max_results": 3,
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data if include_meta else data.get("answer", "")
    except Exception as e:
        print(f"[ERROR] Tavily API failed: {e}")
        return None

