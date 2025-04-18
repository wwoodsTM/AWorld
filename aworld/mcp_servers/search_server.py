"""
Search MCP Server

This module provides MCP server functionality for performing web searches using various search engines.
It supports structured queries and returns formatted search results.

Key features:
- Perform web searches using Exa, Google, and DuckDuckGo
- Filter and format search results
- Validate and process search queries

Main functions:
- mcpsearchexa: Searches the web using Exa
- mcpsearchgoogle: Searches the web using Google
- mcpsearchduckduckgo: Searches the web using DuckDuckGo
"""

import os
import re
import traceback
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
from exa_py import Exa
from exa_py.api import SearchResponse
from pydantic import BaseModel, Field, field_validator

from aworld.logs.util import logger
from aworld.mcp_servers.utils import parse_port, run_mcp_server


# Base search result model that all providers will use
class SearchResult(BaseModel):
    """Base search result model with common fields"""

    id: str
    title: str
    url: str
    snippet: str
    source: str  # Which search engine provided this result

    @field_validator("url")
    def validate_url(cls, v):
        if not v.startswith(("http://", "https://")):
            raise ValueError("Invalid URL format")
        return v


class ExaSearchResult(SearchResult):
    """Exa-specific search result model"""

    publishedDate: str = ""
    author: str = ""
    score: str = ""
    text: str = ""
    image: str = ""
    favicon: str = ""


class GoogleSearchResult(SearchResult):
    """Google-specific search result model"""

    displayLink: str = ""
    formattedUrl: str = ""
    htmlSnippet: str = ""
    htmlTitle: str = ""
    kind: str = ""
    link: str = ""


class DuckDuckGoSearchResult(SearchResult):
    """DuckDuckGo-specific search result model"""

    description: str = ""
    icon: str = ""
    published: str = ""


class SearchResponse(BaseModel):
    """Unified search response model"""

    query: str
    results: List[SearchResult]
    count: int
    source: str
    error: Optional[str] = None


class SearchServer:
    """
    Search Server class for interacting with various search engines.

    This class provides methods for searching the web using Exa, Google, and DuckDuckGo.
    """

    _instance = None
    _exa = None

    def __new__(cls):
        """Implement singleton pattern"""
        if cls._instance is None:
            cls._instance = super(SearchServer, cls).__new__(cls)
            cls._instance._init_server()
        return cls._instance

    def _init_server(self):
        """Initialize the Search server and clients"""
        # Initialize Exa client if API key is available
        api_key = os.environ.get("EXA_API_KEY")
        if api_key:
            self._exa = Exa(api_key=api_key)
            logger.info("Exa client initialized")
        else:
            logger.warning("EXA_API_KEY not found, Exa search will not be available")

        logger.info("SearchServer initialized")

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of SearchServer"""
        if cls._instance is None:
            return cls()
        return cls._instance

    @staticmethod
    def handle_error(e: Exception, operation_type: str) -> str:
        """Unified error handling and return standard format error message"""
        error_msg = f"{operation_type} error: {str(e)}"
        logger.error(f"{operation_type} operation failed: {str(e)}")
        logger.error(traceback.format_exc())

        error_response = SearchResponse(
            query="",
            results=[],
            count=0,
            source=operation_type.lower(),
            error=error_msg,
        )

        return error_response.model_dump_json()

    @classmethod
    def search_exa(
        cls,
        query: str = Field(..., description="The query string."),
        num_results: int = Field(
            20, description="Number of search results to return (default 20)."
        ),
        include_domains: List[str] = Field(
            None, description="Domains to include in the search."
        ),
        exclude_domains: List[str] = Field(
            None, description="Domains to exclude from the search."
        ),
        start_crawl_date: str = Field(
            None, description="Only links crawled after this date."
        ),
        end_crawl_date: str = Field(
            None, description="Only links crawled before this date."
        ),
        start_published_date: str = Field(
            None, description="Only links published after this date."
        ),
        end_published_date: str = Field(
            None, description="Only links published before this date."
        ),
        include_text: List[str] = Field(
            None, description="Strings that must appear in the page text."
        ),
        exclude_text: List[str] = Field(
            None, description="Strings that must not appear in the page text."
        ),
        use_autoprompt: bool = Field(
            False, description="Convert query to Exa (default False)."
        ),
        type: str = Field(
            "neural", description="'keyword' or 'neural' (default 'neural')."
        ),
        category: str = Field(
            "",
            description="available options: (company, research paper, news, pdf, github, tweet, personal site, linkedin profile, financial report)",
        ),
        flags: List[str] = Field(None, description="Experimental flags for Exa usage."),
        moderation: bool = Field(
            False,
            description="If True, the search results will be moderated for safety.",
        ),
        text: bool = Field(
            False, description="Whether to include webpage contents in results."
        ),
    ) -> str:
        """
        Search the web using Exa with a query to retrieve relevant results.

        Args:
            query: The query string
            num_results: Number of search results to return
            include_domains: Domains to include in the search
            exclude_domains: Domains to exclude from the search
            start_crawl_date: Only links crawled after this date
            end_crawl_date: Only links crawled before this date
            start_published_date: Only links published after this date
            end_published_date: Only links published before this date
            include_text: Strings that must appear in the page text
            exclude_text: Strings that must not appear in the page text
            use_autoprompt: Convert query to Exa
            type: 'keyword' or 'neural'
            category: Search category
            flags: Experimental flags for Exa usage
            moderation: If True, the search results will be moderated for safety
            text: Whether to include webpage contents in results

        Returns:
            JSON string containing search results
        """
        try:
            # Handle Field objects if they're passed directly
            if hasattr(query, "default") and not isinstance(query, str):
                query = query.default

            if hasattr(num_results, "default") and not isinstance(num_results, int):
                num_results = num_results.default

            # Get the singleton instance
            instance = cls.get_instance()

            # Check if Exa client is initialized
            if not instance._exa:
                raise ValueError("EXA_API_KEY environment variable not set")

            if type and type not in ["keyword", "neural"]:
                raise ValueError("Search type must be either 'keyword' or 'neural'")

            if start_published_date and end_published_date:
                if start_published_date > end_published_date:
                    raise ValueError(
                        "start_published_date cannot be later than end_published_date"
                    )

            logger.info(f"Exa search starts for query: {query}")
            search_results = [
                ExaSearchResult(
                    id=result_with_text.id,
                    title=result_with_text.title or "",
                    url=result_with_text.url or "",
                    snippet=(
                        result_with_text.text[:100] + "..."
                        if result_with_text.text
                        else ""
                    ),
                    source="exa",
                    publishedDate=result_with_text.published_date or "",
                    author=result_with_text.author or "",
                    score=str(result_with_text.score),
                    text=result_with_text.text or "",
                    image=result_with_text.image or "",
                    favicon=result_with_text.favicon or "",
                )
                for result_with_text in instance._exa.search_and_contents(
                    query,
                    num_results=num_results,
                    include_domains=include_domains,
                    exclude_domains=exclude_domains,
                    start_crawl_date=start_crawl_date,
                    end_crawl_date=end_crawl_date,
                    start_published_date=start_published_date,
                    end_published_date=end_published_date,
                    include_text=include_text,
                    exclude_text=exclude_text,
                    use_autoprompt=use_autoprompt,
                    type=type,
                    category=category,
                    flags=flags,
                    moderation=moderation,
                    text=text,
                ).results
            ]
            search_response = SearchResponse(
                query=query,
                results=search_results,
                count=len(search_results),
                source="exa",
            )
            return search_response.model_dump_json()

        except Exception as e:
            return cls.handle_error(e, "Exa Search")

    @classmethod
    def search_google(
        cls,
        query: str = Field(..., description="The search query string."),
        num_results: int = Field(
            10, description="Number of search results to return (default 10)."
        ),
        safe_search: bool = Field(
            True, description="Whether to enable safe search filtering."
        ),
        language: str = Field("en", description="Language code for search results."),
        country: str = Field("us", description="Country code for search results."),
    ) -> str:
        """
        Search the web using Google Custom Search API.

        Args:
            query: The search query string
            num_results: Number of search results to return
            safe_search: Whether to enable safe search filtering
            language: Language code for search results
            country: Country code for search results

        Returns:
            JSON string containing search results
        """
        try:
            # Handle Field objects if they're passed directly
            if hasattr(query, "default") and not isinstance(query, str):
                query = query.default

            if hasattr(num_results, "default") and not isinstance(num_results, int):
                num_results = num_results.default

            if hasattr(safe_search, "default") and not isinstance(safe_search, bool):
                safe_search = safe_search.default

            if hasattr(language, "default") and not isinstance(language, str):
                language = language.default

            if hasattr(country, "default") and not isinstance(country, str):
                country = country.default

            api_key = os.environ.get("GOOGLE_API_KEY")
            cse_id = os.environ.get("GOOGLE_CSE_ID")

            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
            if not cse_id:
                raise ValueError("GOOGLE_CSE_ID environment variable not set")

            # Ensure num_results is within valid range
            num_results = max(1, num_results)

            # Build the Google Custom Search API URL
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": api_key,
                "cx": cse_id,
                "q": query,
                "num": num_results,
                "safe": "active" if safe_search else "off",
                "hl": language,
                "gl": country,
            }

            logger.info(f"Google search starts for query: {query}")
            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            search_results = []

            if "items" in data:
                for i, item in enumerate(data["items"]):
                    result = GoogleSearchResult(
                        id=f"google-{i}",
                        title=item.get("title", ""),
                        url=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                        source="google",
                        displayLink=item.get("displayLink", ""),
                        formattedUrl=item.get("formattedUrl", ""),
                        htmlSnippet=item.get("htmlSnippet", ""),
                        htmlTitle=item.get("htmlTitle", ""),
                        kind=item.get("kind", ""),
                        link=item.get("link", ""),
                    )
                    search_results.append(result)

            return SearchResponse(
                query=query,
                results=search_results,
                count=len(search_results),
                source="google",
            ).model_dump_json()

        except Exception as e:
            return cls.handle_error(e, "Google Search")

    @classmethod
    def search_duckduckgo(
        cls,
        query: str = Field(..., description="The search query string."),
        num_results: int = Field(
            20, description="Number of search results to return (default 20)."
        ),
        region: str = Field(
            "wt-wt", description="Region code for search results (default: wt-wt)."
        ),
        safe_search: bool = Field(
            True, description="Whether to enable safe search filtering."
        ),
        time_period: str = Field(
            "", description="Time period for results (d, w, m, y)."
        ),
    ) -> str:
        """
        Search the web using DuckDuckGo API.

        Args:
            query: The search query string
            num_results: Number of search results to return
            region: Region code for search results
            safe_search: Whether to enable safe search filtering
            time_period: Time period for results (d, w, m, y)

        Returns:
            JSON string containing search results
        """
        try:
            # Handle Field objects if they're passed directly
            if hasattr(query, "default") and not isinstance(query, str):
                query = query.default

            if hasattr(num_results, "default") and not isinstance(num_results, int):
                num_results = num_results.default

            if hasattr(region, "default") and not isinstance(region, str):
                region = region.default

            if hasattr(safe_search, "default") and not isinstance(safe_search, bool):
                safe_search = safe_search.default

            if hasattr(time_period, "default") and not isinstance(time_period, str):
                time_period = time_period.default

            # Build the DuckDuckGo search URL
            url = "https://html.duckduckgo.com/html/"

            # Prepare parameters
            params = {
                "q": query,
                "kl": region,  # Region/locale
                "kp": "1" if safe_search else "-1",  # Safe search (1 = on, -1 = off)
            }

            # Add time period if specified
            if time_period:
                if time_period in ["d", "w", "m", "y"]:
                    params["df"] = time_period
                else:
                    logger.warning(
                        f"Invalid time period: {time_period}. Using default."
                    )

            # Set headers to mimic a browser
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Referer": "https://duckduckgo.com/",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }

            logger.info(f"DuckDuckGo search starts for query: {query}")
            response = requests.post(url, data=params, headers=headers)
            response.raise_for_status()

            # Parse the HTML response
            soup = BeautifulSoup(response.text, "html.parser")
            results_elements = soup.select(".result")

            search_results = []
            for i, result in enumerate(results_elements[:num_results]):
                # Extract result information
                title_element = result.select_one(".result__a")
                snippet_element = result.select_one(".result__snippet")
                url_element = result.select_one(".result__url")

                if title_element and url_element:
                    title = title_element.get_text(strip=True)
                    raw_url = title_element.get("href", "")

                    # Extract the actual URL from DuckDuckGo's redirect URL
                    url = raw_url
                    if raw_url.startswith("/"):
                        url_match = re.search(r"uddg=([^&]+)", raw_url)
                        if url_match:
                            url = requests.utils.unquote(url_match.group(1))

                    snippet = (
                        snippet_element.get_text(strip=True) if snippet_element else ""
                    )
                    display_url = (
                        url_element.get_text(strip=True) if url_element else ""
                    )

                    result = DuckDuckGoSearchResult(
                        id=f"ddg-{i}",
                        title=title,
                        url=url,
                        snippet=snippet,
                        source="duckduckgo",
                        description=snippet,
                        icon="",
                        published="",
                    )
                    search_results.append(result)

            return SearchResponse(
                query=query,
                results=search_results,
                count=len(search_results),
                source="duckduckgo",
            ).model_dump_json()

        except Exception as e:
            return cls.handle_error(e, "DuckDuckGo Search")


if __name__ == "__main__":
    port = parse_port()

    search_server = SearchServer.get_instance()
    logger.info("RedditServer initialized and ready to handle requests")

    run_mcp_server(
        "Search Server",
        funcs=[
            search_server.search_google,
            # search_server.search_duckduckgo,
            # search_server.search_exa,
        ],
        port=port,
    )
