"""
Module for performing web searches using Google Custom Search.
"""
import requests
from datetime import datetime
import streamlit as st
from typing import List, Dict

class SearchResult:
    """Class to represent a search result with metadata."""
    def __init__(self, title, snippet, url, source="web_search"):
        self.title = title
        self.snippet = snippet
        self.url = url
        self.source = source
        self.timestamp = datetime.now().isoformat()
    
    def to_context_item(self):
        """Format as a context item for the LLM."""
        return {
            "content": f"{self.title}\n{self.snippet}",
            "metadata": {
                "source": self.url,
                "type": "web_search",
                "timestamp": self.timestamp
            },
            "source": self.url
        }

def google_search(query: str, api_key: str, cse_id: str = "876484991592c493d", num_results: int = 5) -> List[SearchResult]:
    """Perform a Google search using the Custom Search JSON API with your CSE ID."""
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cse_id,  # Your CSE ID from the script
        "q": query,
        "num": min(num_results, 10)  # API limit is 10 per request
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        search_results = []
        data = response.json()
        
        if "items" in data:
            for item in data["items"]:
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                url = item.get("link", "")
                search_results.append(SearchResult(title, snippet, url))
        
        return search_results
    
    except Exception as e:
        st.error(f"Error performing Google search: {str(e)}")
        return []

def fetch_page_content(url: str, max_chars: int = 3000) -> str:
    """Fetch and extract text content from a webpage."""
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        response.raise_for_status()
        
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text content
        text = soup.get_text(separator=" ", strip=True)
        
        # Truncate to max_chars
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        
        return text
    
    except Exception as e:
        return f"Error fetching content: {str(e)}"

def enhance_with_web_search(query: str, kb_results: List[Dict]) -> List[Dict]:
    """Enhance context with web search results."""
    # Get Google API key from Streamlit secrets
    if "GOOGLE_API_KEY" not in st.secrets:
        st.warning("Google API key not found in secrets. Web search disabled.")
        return kb_results
    
    # Perform the search
    search_results = google_search(
        query, 
        st.secrets["GOOGLE_API_KEY"], 
        cse_id=st.secrets["GOOGLE_CSE_ID"]
    )
    
    # Convert search results to context items
    web_context = [result.to_context_item() for result in search_results]
    
    # For the top 2 results, try to fetch their full content
    for i, result in enumerate(search_results[:2]):
        if i < len(web_context):
            content = fetch_page_content(result.url)
            if content:
                web_context[i]["content"] += f"\n\nFull content excerpt:\n{content}"
    
    # Combine KB results with web results
    time_indicators = ["recent", "latest", "news", "current", "today"]
    time_sensitive = any(indicator in query.lower() for indicator in time_indicators)
    
    if time_sensitive:
        return web_context + kb_results
    else:
        return kb_results + web_context