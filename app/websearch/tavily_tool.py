import os
import requests
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def search_web(query, include_meta=False):
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.error("Tavily API key not found in .env")
        return None

    url = "https://api.tavily.com/search"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "query": query,
        "search_depth": "basic",
        "include_answer": True,
        "max_results": 3,
    }

    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.2, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))

    try:
        logger.info(f"[Tavily] Sending query: {query}")
        start_time = time.time()
        response = session.post(url, json=payload, headers=headers)
        response.raise_for_status()
        duration = round(time.time() - start_time, 2)

        data = response.json()
        results = data.get("results", [])
        logger.info(f"[Tavily] Query returned {len(results)} results in {duration}s")

        for i, res in enumerate(results):
            title = res.get("title", "No Title")
            url = res.get("url", "No URL")
            logger.info(f"[Tavily] Result {i + 1}: {title} — {url}")

        return data if include_meta else data.get("answer", "")

    except requests.exceptions.HTTPError as http_err:
        logger.error(f"[Tavily] HTTP error {response.status_code}: {response.text}")
    except Exception as e:
        logger.error(f"[Tavily] API request failed: {e}")

    return None
