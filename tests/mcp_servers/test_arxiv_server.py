"""
Unit tests for the ArxivServer class
"""

import json
import os
import shutil
import tempfile
import unittest
from typing import Any, Dict

from aworld.mcp_servers.arxiv_server import ArxivSearchResult, ArxivServer


class TestArxivServer(unittest.TestCase):
    """Test cases for ArxivServer class"""

    def setUp(self):
        """Set up test fixtures"""
        # Reset the singleton instance before each test
        ArxivServer._instance = None
        ArxivServer._client = None

        self.arxiv_server: ArxivServer = ArxivServer.get_instance()

        self.url: str = "https://arxiv.org/abs/2501.09686"
        self.article_id: str = "2501.09686"
        self.title: str = (
            "Towards Large Reasoning Models: A Survey of Reinforced Reasoning with Large Language Models"
        )

    def test_search_by_query(self):
        """Test search_by_query method"""
        paper_info: str = self.arxiv_server.search_arxiv_paper_by_title_or_ids(
            article_title=self.title
        )
        model: ArxivSearchResult = ArxivSearchResult.model_validate_json(paper_info)
        self.assertIsInstance(model, ArxivSearchResult)
        self.assertGreater(len(model.articles), 0)
        self.assertTrue(any(self.title == article.title for article in model.articles))
        self.assertTrue(
            any(self.article_id in article.entry_id for article in model.articles)
        )

    def test_search_by_article_id(self):
        """Test search_by_article_id method"""
        paper_info: str = self.arxiv_server.search_arxiv_paper_by_title_or_ids(
            article_title="", article_ids=[self.article_id]
        )
        model: ArxivSearchResult = ArxivSearchResult.model_validate_json(paper_info)
        self.assertIsInstance(model, ArxivSearchResult)
        self.assertGreater(len(model.articles), 0)
        self.assertTrue(any(self.title == article.title for article in model.articles))
        self.assertTrue(
            any(self.article_id in article.entry_id for article in model.articles)
        )

    def test_download_by_article_id(self):
        """Test download_by_article_id method"""
        temp_dir = tempfile.mkdtemp()

        try:
            download_info: str = self.arxiv_server.download_arxiv_paper(
                article_ids=[self.article_id], output_dir=temp_dir
            )
            # Parse the JSON response
            download_data: Dict[str, Any] = json.loads(download_info)
            # {
            #     "total": len(article_ids),
            #     "success_count": sum(1 for r in download_results if r.success),
            #     "failed_count": sum(1 for r in download_results if not r.success),
            #     "results": [r.model_dump() for r in download_results],
            # }
            self.assertEqual(download_data["total"], 1)
            self.assertGreater(download_data["success_count"], 0)
            self.assertEqual(download_data["failed_count"], 0)
            self.assertEqual(len(download_data["results"]), 1)
            # ArxivDownloadResult(
            #     entry_id=article.entry_id,
            #     title=article.title,
            #     file_path=file_path,
            #     file_size=file_size,
            #     success=True,
            #     error=None,
            # )
            self.assertTrue(self.article_id in download_data["results"][0]["entry_id"])
            self.assertEqual(download_data["results"][0]["title"], self.title)
            self.assertGreater(download_data["results"][0]["file_size"], 0)
            self.assertTrue(download_data["results"][0]["success"])
            self.assertIsNone(download_data["results"][0]["error"])
            # Verify the file exists in the temp directory
            self.assertTrue(os.path.exists(download_data["results"][0]["file_path"]))
            self.assertTrue(os.path.isfile(download_data["results"][0]["file_path"]))
            self.assertTrue(download_data["results"][0]["file_path"].endswith(".pdf"))
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
