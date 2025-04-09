import asyncio

from dotenv import load_dotenv

from aworld.agents.debate.base import DebateSpeech
from aworld.agents.debate.search.tavily_search_engine import TavilySearchEngine


class SearchResult:
    id: str
    url: str
    title: str
    content: str


async def deepsearch(llm, topic, opinion, oppose_opinion, last_oppose_speech_content: str,
                     history: list[DebateSpeech]) -> list[SearchResult]:
    search_engine = TavilySearchEngine()

    results = search_engine.async_batch_search(queries=["杭州天气怎么样", "xxx"], max_results=5)

    pass

if __name__ == '__main__':
    load_dotenv()
    search_engine = TavilySearchEngine()
    results = asyncio.run(search_engine.async_batch_search(queries=["杭州天气怎么样", "xxx"], max_results=5))
    print(results)