"""
Web Search Tool - Search the internet for current information.
Uses Tavily API for high-quality search results.
"""

import asyncio
from typing import Any, Dict, List
import aiohttp
import json
from urllib.parse import quote

from tools import BaseTool


class WebSearchTool(BaseTool):
    """
    Search the web for current information.
    Useful for answering questions about current events, facts, news, etc.
    Uses Tavily API for reliable, high-quality search results.
    Get API key from: https://tavily.com
    """
    
    def _get_description(self) -> str:
        return """
        Search the internet for current information, news, facts, and answers.
        Use this tool when you need to find information that might not be in your training data,
        such as current events, weather forecasts, news, stock prices, sports scores, etc.
        
        Examples:
        - "What is the latest news about AI?"
        - "Weather forecast for New York"
        - "Current stock price of Tesla"
        - "Who won the latest Super Bowl?"
        """
    
    def _get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of search results to return (1-10)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.config = config
        self.tavily_api_key = config.get('web_search', {}).get('tavily_api_key') or config.get('tavily', {}).get('api_key')
    
    async def execute(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """Execute web search."""
        print(f"🔍 Searching web for: '{query}'")
        
        try:
            # Try Tavily API first if key is available
            if self.tavily_api_key:
                results = await self._search_tavily(query, num_results)
                if results:
                    return {
                        "success": True,
                        "query": query,
                        "results_count": len(results),
                        "results": results,
                        "source": "tavily"
                    }
            
            # Fallback to DuckDuckGo if Tavily fails or no API key
            results = await self._search_duckduckgo(query, num_results)
            
            if results:
                return {
                    "success": True,
                    "query": query,
                    "results_count": len(results),
                    "results": results,
                    "source": "duckduckgo"
                }
            else:
                return {
                    "success": False,
                    "error": "No results found",
                    "query": query
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    async def _search_tavily(self, query: str, num_results: int = 5) -> List[Dict]:
        """Search using Tavily API."""
        from tavily import TavilyClient
        
        results = []
        
        try:
            client = TavilyClient(api_key=self.tavily_api_key)
            
            # Perform search
            response = client.search(
                query=query,
                max_results=num_results,
                search_depth="basic",  # Options: basic, advanced
                include_answer=False,
                include_raw_content=False
            )
            
            # Extract results
            for result in response.get("results", []):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("content", ""),
                    "score": result.get("score", 0)
                })
            
            return results
            
        except Exception as e:
            print(f"⚠️  Tavily search failed: {e}")
            return []
    
    async def _search_duckduckgo(self, query: str, num_results: int = 5) -> List[Dict]:
        """Search using DuckDuckGo HTML interface (fallback)."""
        from ddgs import DDGS
        
        results = []
        
        try:
            with DDGS() as ddgs:
                search_results = ddgs.text(query, max_results=num_results)
                
                for result in search_results:
                    results.append({
                        "title": result.get("title", ""),
                        "url": result.get("href", ""),
                        "snippet": result.get("body", "")
                    })
        except Exception as e:
            print(f"⚠️  DuckDuckGo search failed: {e}")
        
        return results


class WebNewsTool(BaseTool):
    """
    Search for news articles on the web.
    Uses Tavily API for high-quality news results.
    """
    
    def _get_description(self) -> str:
        return """
        Search for current news articles on the internet.
        Use this to find the latest news on any topic.
        
        Examples:
        - "Latest news about artificial intelligence"
        - "Breaking news today"
        - "Tech news this week"
        """
    
    def _get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "News topic to search for"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of news articles to return (1-10)",
                    "default": 5
                }
            },
            "required": ["topic"]
        }
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.config = config
        self.tavily_api_key = config.get('web_search', {}).get('tavily_api_key') or config.get('tavily', {}).get('api_key')
    
    async def execute(self, topic: str, num_results: int = 5) -> Dict[str, Any]:
        """Execute news search."""
        print(f"📰 Searching news for: '{topic}'")
        
        try:
            # Try Tavily API first if key is available
            if self.tavily_api_key:
                results = await self._search_news_tavily(topic, num_results)
                if results:
                    return {
                        "success": True,
                        "topic": topic,
                        "results_count": len(results),
                        "results": results,
                        "source": "tavily"
                    }
            
            # Fallback to DuckDuckGo if Tavily fails or no API key
            results = await self._search_news_duckduckgo(topic, num_results)
            
            if results:
                return {
                    "success": True,
                    "topic": topic,
                    "results_count": len(results),
                    "results": results,
                    "source": "duckduckgo"
                }
            else:
                return {
                    "success": False,
                    "error": "No news found",
                    "topic": topic
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "topic": topic
            }
    
    async def _search_news_tavily(self, topic: str, num_results: int = 5) -> List[Dict]:
        """Search for news using Tavily API."""
        from tavily import TavilyClient
        
        results = []
        
        try:
            client = TavilyClient(api_key=self.tavily_api_key)
            
            # Add "news" or "latest" to topic for better news results
            search_query = f"latest news {topic}"
            
            # Perform search
            response = client.search(
                query=search_query,
                max_results=num_results,
                search_depth="basic",
                include_answer=False,
                include_raw_content=False
            )
            
            # Extract results
            for result in response.get("results", []):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "source": result.get("source", "Web"),
                    "date": "Recent",  # Tavily doesn't provide exact dates
                    "snippet": result.get("content", ""),
                    "score": result.get("score", 0)
                })
            
            return results
            
        except Exception as e:
            print(f"⚠️  Tavily news search failed: {e}")
            return []
    
    async def _search_news_duckduckgo(self, topic: str, num_results: int = 5) -> List[Dict]:
        """Search for news using DuckDuckGo (fallback)."""
        from ddgs import DDGS
        
        results = []
        
        try:
            with DDGS() as ddgs:
                search_results = ddgs.news(topic, max_results=num_results)
                
                for result in search_results:
                    results.append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "source": result.get("source", ""),
                        "date": result.get("date", ""),
                        "snippet": result.get("body", "")
                    })
        except Exception as e:
            print(f"⚠️  DuckDuckGo news search failed: {e}")
        
        return results
