"""
Wayback Machine MCP Server

This module provides a microservice for interacting with the Internet Archive's Wayback Machine through MCP.
It enables retrieving archived web pages, checking availability of URLs in the archive,
and searching for archived snapshots with proper error handling.

Key features:
- Retrieve archived web pages from specific dates
- Check if a URL is archived in the Wayback Machine
- Get available snapshots for a URL
- Search for archived content
- Download archived files

Main functions:
- get_snapshot: Retrieves an archived web page from a specific date
- check_availability: Checks if a URL is available in the Wayback Machine
- get_snapshots: Gets a list of available snapshots for a URL
- search_wayback: Searches for archived content matching criteria
- download_file: Downloads an archived file
"""

import json
import os
import traceback
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from aworld.logs.util import logger
from aworld.mcp_servers.utils import parse_port, run_mcp_server
from aworld.utils import import_package


# Define model classes for different Wayback Machine API responses
class WaybackSnapshot(BaseModel):
    """Model representing a Wayback Machine snapshot"""

    url: str
    timestamp: str
    status_code: Optional[int] = None
    mime_type: Optional[str] = None
    snapshot_url: str
    archived_date: str


class WaybackAvailability(BaseModel):
    """Model representing Wayback Machine availability for a URL"""

    url: str
    is_available: bool
    first_snapshot: Optional[str] = None
    last_snapshot: Optional[str] = None
    total_snapshots: int = 0


class WaybackSearchResult(BaseModel):
    """Model representing search results from Wayback Machine"""

    query: str
    total_results: int
    results: List[WaybackSnapshot] = []


class WaybackContent(BaseModel):
    """Model representing content retrieved from Wayback Machine"""

    url: str
    timestamp: str
    content: str
    content_type: str
    snapshot_url: str
    status_code: Optional[int] = None


class WaybackError(BaseModel):
    """Model representing an error in Wayback Machine API processing"""

    error: str
    operation: str


class WaybackServer:
    """
    Wayback Machine Server class for interacting with the Internet Archive's Wayback Machine.

    This class provides methods for retrieving archived web pages, checking availability,
    and searching for archived snapshots.
    """

    _instance = None
    _wayback = None

    def __new__(cls):
        """Implement singleton pattern"""
        if cls._instance is None:
            cls._instance = super(WaybackServer, cls).__new__(cls)
            cls._instance._init_server()
        return cls._instance

    def _init_server(self):
        """Initialize the Wayback Machine server and client"""
        # Import wayback-machine-scraper package, install if not available
        import_package("waybackpy", install_name="waybackpy")
        from waybackpy import WaybackMachineAvailabilityAPI, WaybackMachineCDXServerAPI

        self._wayback_cdx = WaybackMachineCDXServerAPI
        self._wayback_availability = WaybackMachineAvailabilityAPI

        logger.info("WaybackServer initialized")

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of WaybackServer"""
        if cls._instance is None:
            return cls()
        return cls._instance

    @staticmethod
    def handle_error(e: Exception, operation_type: str) -> str:
        """Unified error handling and return standard format error message"""
        error_msg = f"{operation_type} error: {str(e)}"
        logger.error(traceback.format_exc())

        error = WaybackError(error=error_msg, operation=operation_type)

        return error.model_dump_json()

    @staticmethod
    def _format_timestamp(timestamp: str) -> str:
        """
        Format timestamp to human-readable date.

        Args:
            timestamp: Wayback Machine timestamp (YYYYMMDDhhmmss format)

        Returns:
            Formatted date string (YYYY-MM-DD HH:MM:SS)
        """
        try:
            dt = datetime.strptime(timestamp, "%Y%m%d%H%M%S")
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            return timestamp

    @classmethod
    def get_snapshot(
        cls,
        url: str = Field(description="URL to retrieve from archive"),
        timestamp: str = Field(
            default="",
            description="Timestamp in YYYYMMDDhhmmss format or 'earliest'/'latest'",
        ),
        user_agent: str = Field(
            default="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            description="User agent string for the request",
        ),
    ) -> str:
        """
        Get a snapshot of a URL from the Wayback Machine.

        Args:
            url: URL to retrieve from archive
            timestamp: Timestamp in YYYYMMDDhhmmss format or 'earliest'/'latest'
            user_agent: User agent string for the request

        Returns:
            JSON string containing the snapshot content and metadata
        """
        try:
            # Handle Field objects if they're passed directly
            if hasattr(url, "default") and not isinstance(url, str):
                url = url.default

            if hasattr(timestamp, "default") and not isinstance(timestamp, str):
                timestamp = timestamp.default

            if hasattr(user_agent, "default") and not isinstance(user_agent, str):
                user_agent = user_agent.default

            # Get the singleton instance
            instance = cls.get_instance()

            # Import necessary classes
            from waybackpy import WaybackMachineAvailabilityAPI

            # Create availability API instance
            wayback = WaybackMachineAvailabilityAPI(url=url, user_agent=user_agent)

            # Get the snapshot
            if timestamp.lower() == "earliest":
                snapshot = wayback.earliest()
            elif timestamp.lower() == "latest":
                snapshot = wayback.newest()
            elif timestamp:
                # Use specific timestamp
                from waybackpy import WaybackMachineSaveAPI

                save_api = WaybackMachineSaveAPI(
                    url=url, user_agent=user_agent, timestamp=timestamp
                )
                snapshot = save_api.get_archive()
            else:
                # Default to newest
                snapshot = wayback.newest()

            # Get content
            content = snapshot.text

            # Create content object
            content_obj = WaybackContent(
                url=url,
                timestamp=snapshot.timestamp,
                content=content,
                content_type=snapshot.headers.get("Content-Type", "text/html"),
                snapshot_url=snapshot.archive_url,
                status_code=snapshot.status_code,
            )

            return content_obj.model_dump_json()

        except Exception as e:
            return cls.handle_error(e, "Get Snapshot")

    @classmethod
    def check_availability(
        cls,
        url: str = Field(description="URL to check in the archive"),
        user_agent: str = Field(
            default="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            description="User agent string for the request",
        ),
    ) -> str:
        """
        Check if a URL is available in the Wayback Machine.

        Args:
            url: URL to check
            user_agent: User agent string for the request

        Returns:
            JSON string containing availability information
        """
        try:
            # Handle Field objects if they're passed directly
            if hasattr(url, "default") and not isinstance(url, str):
                url = url.default

            if hasattr(user_agent, "default") and not isinstance(user_agent, str):
                user_agent = user_agent.default

            # Get the singleton instance
            instance = cls.get_instance()

            # Import necessary classes
            from waybackpy import WaybackMachineCDXServerAPI

            # Create CDX API instance
            cdx_api = WaybackMachineCDXServerAPI(url=url, user_agent=user_agent)

            # Get snapshots to check availability
            snapshots = list(cdx_api.snapshots())
            is_available = len(snapshots) > 0

            # Get first and last snapshot timestamps if available
            first_snapshot = None
            last_snapshot = None

            if is_available:
                first_snapshot = cls._format_timestamp(snapshots[0].timestamp)
                last_snapshot = cls._format_timestamp(snapshots[-1].timestamp)

            # Create availability object
            availability = WaybackAvailability(
                url=url,
                is_available=is_available,
                first_snapshot=first_snapshot,
                last_snapshot=last_snapshot,
                total_snapshots=len(snapshots),
            )

            return availability.model_dump_json()

        except Exception as e:
            return cls.handle_error(e, "Check Availability")

    @classmethod
    def get_snapshots(
        cls,
        url: str = Field(description="URL to get snapshots for"),
        from_date: str = Field(default="", description="Start date in YYYYMMDD format"),
        to_date: str = Field(default="", description="End date in YYYYMMDD format"),
        limit: int = Field(
            default=50, description="Maximum number of snapshots to return (max 1000)"
        ),
        user_agent: str = Field(
            default="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            description="User agent string for the request",
        ),
    ) -> str:
        """
        Get a list of available snapshots for a URL.

        Args:
            url: URL to get snapshots for
            from_date: Start date in YYYYMMDD format
            to_date: End date in YYYYMMDD format
            limit: Maximum number of snapshots to return
            user_agent: User agent string for the request

        Returns:
            JSON string containing list of snapshots
        """
        try:
            # Handle Field objects if they're passed directly
            if hasattr(url, "default") and not isinstance(url, str):
                url = url.default

            if hasattr(from_date, "default") and not isinstance(from_date, str):
                from_date = from_date.default

            if hasattr(to_date, "default") and not isinstance(to_date, str):
                to_date = to_date.default

            if hasattr(limit, "default") and not isinstance(limit, int):
                limit = limit.default

            if hasattr(user_agent, "default") and not isinstance(user_agent, str):
                user_agent = user_agent.default

            # Validate input
            if limit > 1000:
                limit = 1000
                logger.warning("Limit capped at 1000 snapshots")

            # Get the singleton instance
            instance = cls.get_instance()

            # Import necessary classes
            from waybackpy import WaybackMachineCDXServerAPI

            # Create CDX API instance
            cdx_api = WaybackMachineCDXServerAPI(
                url=url,
                user_agent=user_agent,
                start_timestamp=from_date if from_date else None,
                end_timestamp=to_date if to_date else None,
            )

            # Get snapshots
            snapshots_iter = cdx_api.snapshots()

            # Process snapshots
            snapshots = []
            count = 0

            for snapshot in snapshots_iter:
                if count >= limit:
                    break

                snapshot_obj = WaybackSnapshot(
                    url=url,
                    timestamp=snapshot.timestamp,
                    status_code=snapshot.status_code,
                    mime_type=snapshot.mime_type,
                    snapshot_url=snapshot.archive_url,
                    archived_date=cls._format_timestamp(snapshot.timestamp),
                )

                snapshots.append(snapshot_obj)
                count += 1

            # Create result
            result = {
                "url": url,
                "snapshots": [snapshot.model_dump() for snapshot in snapshots],
                "count": len(snapshots),
                "from_date": from_date if from_date else "earliest",
                "to_date": to_date if to_date else "latest",
            }

            return json.dumps(result)

        except Exception as e:
            return cls.handle_error(e, "Get Snapshots")

    @classmethod
    def search_wayback(
        cls,
        query: str = Field(description="Search query"),
        match_type: str = Field(
            default="exact",
            description="Match type: 'exact', 'prefix', 'host', 'domain'",
        ),
        limit: int = Field(
            default=50, description="Maximum number of results to return (max 1000)"
        ),
        user_agent: str = Field(
            default="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            description="User agent string for the request",
        ),
    ) -> str:
        """
        Search for archived content in the Wayback Machine.

        Args:
            query: Search query (URL or domain)
            match_type: Match type ('exact', 'prefix', 'host', 'domain')
            limit: Maximum number of results to return
            user_agent: User agent string for the request

        Returns:
            JSON string containing search results
        """
        try:
            # Handle Field objects if they're passed directly
            if hasattr(query, "default") and not isinstance(query, str):
                query = query.default

            if hasattr(match_type, "default") and not isinstance(match_type, str):
                match_type = match_type.default

            if hasattr(limit, "default") and not isinstance(limit, int):
                limit = limit.default

            if hasattr(user_agent, "default") and not isinstance(user_agent, str):
                user_agent = user_agent.default

            # Validate input
            if limit > 1000:
                limit = 1000
                logger.warning("Limit capped at 1000 results")

            # Get the singleton instance
            instance = cls.get_instance()

            # Import necessary classes
            from waybackpy import WaybackMachineCDXServerAPI

            # Create CDX API instance with appropriate match type
            cdx_api = WaybackMachineCDXServerAPI(
                url=query, user_agent=user_agent, match_type=match_type
            )

            # Get search results
            results_iter = cdx_api.snapshots()

            # Process results
            results = []
            count = 0

            for result in results_iter:
                if count >= limit:
                    break

                result_obj = WaybackSnapshot(
                    url=result.original_url,
                    timestamp=result.timestamp,
                    status_code=result.status_code,
                    mime_type=result.mime_type,
                    snapshot_url=result.archive_url,
                    archived_date=cls._format_timestamp(result.timestamp),
                )

                results.append(result_obj)
                count += 1

            # Create search result
            search_result = WaybackSearchResult(
                query=query,
                total_results=count,  # We can only count what we've seen
                results=[result.model_dump() for result in results],
            )

            return search_result.model_dump_json()

        except Exception as e:
            return cls.handle_error(e, "Search Wayback")

    @classmethod
    def download_file(
        cls,
        url: str = Field(description="URL to download from archive"),
        timestamp: str = Field(
            default="",
            description="Timestamp in YYYYMMDDhhmmss format or 'earliest'/'latest'",
        ),
        output_format: str = Field(
            default="base64", description="Output format: 'base64', 'binary', 'text'"
        ),
        user_agent: str = Field(
            default="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            description="User agent string for the request",
        ),
    ) -> str:
        """
        Download a file from the Wayback Machine.

        Args:
            url: URL to download from archive
            timestamp: Timestamp in YYYYMMDDhhmmss format or 'earliest'/'latest'
            output_format: Output format ('base64', 'binary', 'text')
            user_agent: User agent string for the request

        Returns:
            JSON string containing the file content and metadata
        """
        try:
            # Handle Field objects if they're passed directly
            if hasattr(url, "default") and not isinstance(url, str):
                url = url.default

            if hasattr(timestamp, "default") and not isinstance(timestamp, str):
                timestamp = timestamp.default

            if hasattr(output_format, "default") and not isinstance(output_format, str):
                output_format = output_format.default

            if hasattr(user_agent, "default") and not isinstance(user_agent, str):
                user_agent = user_agent.default

            # Get the singleton instance
            instance = cls.get_instance()

            # Import necessary classes
            import base64

            from waybackpy import WaybackMachineAvailabilityAPI

            # Create availability API instance
            wayback = WaybackMachineAvailabilityAPI(url=url, user_agent=user_agent)

            # Get the snapshot
            if timestamp.lower() == "earliest":
                snapshot = wayback.earliest()
            elif timestamp.lower() == "latest":
                snapshot = wayback.newest()
            elif timestamp:
                # Use specific timestamp
                from waybackpy import WaybackMachineSaveAPI

                save_api = WaybackMachineSaveAPI(
                    url=url, user_agent=user_agent, timestamp=timestamp
                )
                snapshot = save_api.get_archive()
            else:
                # Default to newest
                snapshot = wayback.newest()

            # Get content based on output format
            if output_format.lower() == "base64":
                content = base64.b64encode(snapshot.content).decode("utf-8")
            elif output_format.lower() == "binary":
                content = snapshot.content
            else:  # text
                content = snapshot.text

            # Create content object
            result = {
                "url": url,
                "timestamp": snapshot.timestamp,
                "content": content,
                "content_type": snapshot.headers.get(
                    "Content-Type", "application/octet-stream"
                ),
                "snapshot_url": snapshot.archive_url,
                "status_code": snapshot.status_code,
                "output_format": output_format,
            }

            return json.dumps(result)

        except Exception as e:
            return cls.handle_error(e, "Download File")


# Main function
if __name__ == "__main__":
    port = parse_port()

    wayback_server = WaybackServer.get_instance()
    logger.info("WaybackServer initialized and ready to handle requests")

    run_mcp_server(
        "Wayback Machine Server",
        funcs=[
            wayback_server.get_snapshot,
            wayback_server.check_availability,
            wayback_server.get_snapshots,
            wayback_server.search_wayback,
            wayback_server.download_file,
        ],
        port=port,
    )
