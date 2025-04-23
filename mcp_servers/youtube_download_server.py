import asyncio
import json
from apify_client import ApifyClient
from typing import List
from pydantic import Field
from mcp.server.fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("youtube-download-server")

@mcp.tool(description="Download YouTube video with Apify and return a Downloadable URL.")
async def mcp_download_youtube_video(
    url: str = Field(
        description="The origin URL of the YouTube video to download."
    )
) -> str:
    """
    Download YouTube video using Apify actor and return dataset information.

    Args:
        url (str): The URL of the YouTube video to download.
        max_videos (int): The maximum number of videos to download. Default is 1.

    Returns:
        str: Information about the dataset containing the download results.
    """
    try:
        # Initialize the ApifyClient with your Apify API token
        client = ApifyClient(os.getenv("APIFY_API_TOKEN"))

        # Prepare the Actor input with proper startUrls format
        run_input = {
            "startUrls": [url],  # Pass URL directly as string
            "maxVideos": 1,
            "proxy": {
                "useApifyProxy": True
            }
        }

        # Run the Actor and wait for it to finish
        run = client.actor("epctex/youtube-video-downloader").call(run_input=run_input)

        # Fetch and return Actor results from the run's dataset (if there are any)
        # result_info = "Check your data here: https://console.apify.com/storage/datasets/" + run["defaultDatasetId"]
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            downloadUrl = item["downloadUrl"]
        
        return downloadUrl
    except Exception as e:
        error_result = {
            "error": str(e)
        }
        return json.dumps(error_result, ensure_ascii=False)


def main():
    from dotenv import load_dotenv
    load_dotenv()
    
    print("Starting YouTube Download MCP Server...", file=sys.stderr)
    mcp.run(transport='stdio')

# Make the module callable
def __call__():
    """
    Make the module callable for uvx.
    This function is called when the module is executed directly.
    """
    main()


# Add this for compatibility with uvx
import sys
sys.modules[__name__].__call__ = __call__

# Run the server when the script is executed directly
if __name__ == "__main__":
    main()