"""
Download MCP Server

This module provides MCP server functionality for downloading files from URLs.
It handles various download scenarios with proper validation, error handling,
and progress tracking.

Key features:
- File downloading from HTTP/HTTPS URLs
- Download progress tracking
- File validation
- Safe file saving

Main functions:
- download_files: Downloads files from URLs to local filesystem
"""

import os
import traceback
import urllib.parse
from pathlib import Path
from typing import List, Optional

import requests
from pydantic import BaseModel, Field

from aworld.logs.util import logger
from aworld.mcp_servers.abc.base import MCPServerBase, mcp
from aworld.mcp_servers.utils import parse_port, run_mcp_server


class DownloadResult(BaseModel):
    """Download result model with file information"""

    file_path: str
    file_name: str
    file_size: int
    content_type: Optional[str] = None
    success: bool
    error: Optional[str] = None


class DownloadResults(BaseModel):
    """Download results model for multiple files"""

    results: List[DownloadResult]
    success_count: int
    failed_count: int


class DownloadServer(MCPServerBase):
    """
    Download Server class for downloading files from URLs.

    This class provides methods for downloading files from HTTP/HTTPS URLs
    with proper validation, error handling, and progress tracking.
    """

    _instance = None

    def __new__(cls):
        """Implement singleton pattern"""
        if cls._instance is None:
            cls._instance = super(DownloadServer, cls).__new__(cls)
            cls._instance._init_server()
        return cls._instance

    def _init_server(self):
        """Initialize the download server"""
        logger.info("DownloadServer initialized")

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of DownloadServer"""
        if cls._instance is None:
            return cls()
        return cls._instance

    @mcp
    @classmethod
    def download_files(
        cls,
        urls: List[str] = Field(
            ...,
            description="The URLs of the files to download. Must be a list of URLs.",
        ),
        output_dir: str = Field(
            "/tmp/mcp_downloads",
            description="Directory to save the downloaded files (default: /tmp/mcp_downloads).",
        ),
        timeout: int = Field(
            60, description="Download timeout in seconds (default: 60)."
        ),
    ) -> str:
        """Download files from URLs and save to the local filesystem.

        Args:
            urls: The URLs of the files to download, must be a list of URLs
            output_dir: Directory to save the downloaded files
            timeout: Download timeout in seconds

        Returns:
            JSON string with download results information
        """
        # Handle Field objects if they're passed directly
        if hasattr(urls, "default") and not isinstance(urls, list):
            urls = urls.default

        if hasattr(output_dir, "default") and not isinstance(output_dir, str):
            output_dir = output_dir.default

        if hasattr(timeout, "default") and not isinstance(timeout, int):
            timeout = timeout.default

        results = []
        success_count = 0
        failed_count = 0

        for single_url in urls:
            result_json = cls._download_single_file(single_url, output_dir, "", timeout)
            result = DownloadResult.model_validate_json(result_json)
            results.append(result)

            if result.success:
                success_count += 1
            else:
                failed_count += 1

        batch_results = DownloadResults(
            results=results, success_count=success_count, failed_count=failed_count
        )

        return batch_results.model_dump_json()

    @classmethod
    def _download_single_file(
        cls, url: str, output_dir: str, filename: str, timeout: int
    ) -> str:
        """Download a single file from URL and save to the local filesystem.

        Args:
            url: The URL of the file to download
            output_dir: Directory to save the downloaded file
            filename: Optional filename to use (if empty, extracted from URL)
            timeout: Download timeout in seconds

        Returns:
            JSON string with download result information
        """
        try:
            # Validate URL
            if not url.startswith(("http://", "https://")):
                error_msg = (
                    f"Invalid URL: {url}. Only HTTP and HTTPS URLs are supported."
                )
                return DownloadResult(
                    file_path="",
                    file_name="",
                    file_size=0,
                    success=False,
                    error=error_msg,
                ).model_dump_json()

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Extract filename from URL if not provided
            if not filename:
                parsed_url = urllib.parse.urlparse(url)
                filename = os.path.basename(parsed_url.path)
                if not filename:
                    filename = "downloaded_file"

            # Ensure filename is safe
            filename = Path(filename).name  # Remove any path components

            # Full path to save the file
            file_path = os.path.join(output_dir, filename)

            # Download the file with progress tracking
            logger.info(f"Downloading {url} to {file_path}")
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()

            # Get content type
            content_type = response.headers.get("Content-Type")

            # Get file size
            file_size = int(response.headers.get("Content-Length", 0))

            # Save the file
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # Verify file was saved
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Failed to save file to {file_path}")

            # Get actual file size
            actual_size = os.path.getsize(file_path)

            logger.success(f"Successfully downloaded {url} to {file_path}")

            return DownloadResult(
                file_path=file_path,
                file_name=filename,
                file_size=actual_size,
                content_type=content_type,
                success=True,
            ).model_dump_json()

        except requests.exceptions.RequestException as e:
            error_msg = f"Download error: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return DownloadResult(
                file_path="",
                file_name=filename,
                file_size=0,
                success=False,
                error=error_msg,
            ).model_dump_json()
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return DownloadResult(
                file_path="",
                file_name=filename,
                file_size=0,
                success=False,
                error=error_msg,
            ).model_dump_json()


# Main function
if __name__ == "__main__":
    port = parse_port()

    download_server = DownloadServer.get_instance()
    logger.info("DownloadServer initialized and ready to handle requests")

    run_mcp_server(
        "Download Server",
        funcs=[download_server.download_files],
        port=port,
    )
