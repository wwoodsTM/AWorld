import asyncio
import json
import logging
import os
import sys

import aiohttp
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from mcp.server import FastMCP
from pydantic import Field

from aworld.logs.util import logger


mcp = FastMCP("aworldsearch-server")

async def search_single(query: str, num: int = 5) -> Optional[Dict[str, Any]]:
    """Execute a single search query, returns None on error"""
    try:
        url = os.getenv('AWORLD_SEARCH_URL')
        searchMode = os.getenv('AWORLD_SEARCH_SEARCHMODE')
        source = os.getenv('AWORLD_SEARCH_SOURCE')
        domain = os.getenv('AWORLD_SEARCH_DOMAIN')
        uid = os.getenv('AWORLD_SEARCH_UID')
        if not url or not searchMode or not source or not domain:
            logger.warning(f"Query failed: url, searchMode, source, domain parameters incomplete")
            return None

        headers = {
            'Content-Type': 'application/json'
        }
        data = {
            "domain": domain,
            "extParams": {},
            "page": 0,
            "pageSize": num,
            "query": query,
            "searchMode": searchMode,
            "source": source,
            "userId": uid
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status != 200:
                        logger.warning(f"Query failed: {query}, status code: {response.status}")
                        return None
                    
                    result = await response.json()
                    return result
            except aiohttp.ClientError:
                logger.warning(f"Request error: {query}")
                return None
    except Exception:
        logger.warning(f"Query exception: {query}")
        return None


def filter_valid_docs(result: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter valid document results, returns empty list if input is None"""
    if result is None:
        return []
    
    try:
        valid_docs = []
        
        # Check success field
        if not result.get("success"):
            return valid_docs
        
        # Check searchDocs field
        search_docs = result.get("searchDocs", [])
        if not search_docs:
            return valid_docs
        
        # Extract required fields
        required_fields = ["title", "docAbstract", "url", "doc"]
        
        for doc in search_docs:
            # Check if all required fields exist and are not empty
            is_valid = True
            for field in required_fields:
                if field not in doc or not doc[field]:
                    is_valid = False
                    break
            
            if is_valid:
                # Keep only required fields
                filtered_doc = {field: doc[field] for field in required_fields}
                valid_docs.append(filtered_doc)
        
        return valid_docs
    except Exception:
        return []

@mcp.tool(description="Search based on the user's input query list")
async def search(
    query_list: List[str] = Field(
            description="List format, queries to search for, minimum 1, maximum 5"
        ),
    num: int = Field(
    5,
        description="Maximum number of results per query, default is 5, please keep the total results within 15"
    )
) -> Any:
    """Execute search main function, supports single query or query list"""
    try:
        # Get configuration from environment variables
        env_total_num = os.getenv('AWORLD_SEARCH_TOTAL_NUM')
        if env_total_num and env_total_num.isdigit():
            # Force override input num parameter with environment variable
            num = int(env_total_num)
        
        # If no queries provided, return empty list
        if not query_list:
            return json.dumps([])
        
        # When query count is >= 3 or slice_num is set, use corresponding value
        slice_num = os.getenv('AWORLD_SEARCH_SLICE_NUM')
        if slice_num and slice_num.isdigit():
            actual_num = int(slice_num)
        else:
            actual_num = 2 if len(query_list) >= 3 else num
        
        # Execute all queries in parallel
        tasks = [search_single(q, actual_num) for q in query_list]
        raw_results = await asyncio.gather(*tasks)
        
        # Filter and merge results
        all_valid_docs = []
        for result in raw_results:
            valid_docs = filter_valid_docs(result)
            all_valid_docs.extend(valid_docs)
        
        # Return merged results
        result_json = json.dumps(all_valid_docs, ensure_ascii=False)
        logger.info(f"Completed {len(query_list)} queries, found {len(all_valid_docs)} valid documents")
        logger.info(result_json)
        
        return result_json
    except Exception:
        # Return empty list on exception
        return json.dumps([])


def main():
    from dotenv import load_dotenv

    load_dotenv()

    print("Starting Audio MCP aworldsearch-server...", file=sys.stderr)
    mcp.run(transport="stdio")


# Make the module callable
def __call__():
    """
    Make the module callable for uvx.
    This function is called when the module is executed directly.
    """
    main()

sys.modules[__name__].__call__ = __call__

if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     # Configure logging
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     )
#
#
#     # Test single query
#     # asyncio.run(search("Alibaba financial report"))
#
#     # Test multiple queries
#     test_queries = ["Alibaba financial report", "Tencent financial report", "Baidu financial report"]
#     asyncio.run(search(query_list=test_queries))