"""
Video MCP Server

This module provides MCP server functionality for video processing and analysis.
It handles video frame extraction, analysis, and various video-related tasks such as
subtitle extraction, content summarization, and visual analysis.

The server supports both local video files and remote video URLs with proper validation
and handles various video formats including MP4, AVI, MOV, and others.

Main functions:
- get_video_frames: Extracts frames from videos at specified intervals
- Various MCP tools for video analysis and processing
"""

import base64
import os
import traceback
from typing import Any, Dict, List

import cv2
from pydantic import Field

from aworld.logs.util import logger
from aworld.mcp_servers.abc.base import MCPServerBase, mcp
from aworld.mcp_servers.utils import (
    get_file_from_source,
    get_llm_config_from_os_environ,
    handle_llm_response,
    parse_port,
    run_mcp_server,
)
from aworld.models.llm import get_llm_model


class VideoServer(MCPServerBase):
    """
    Video Server class for processing and analyzing video content.

    This class provides methods for extracting frames from videos, analyzing content,
    extracting subtitles, and summarizing video content.
    """

    _instance = None
    _llm_config = None

    def __new__(cls):
        """Implement singleton pattern"""
        if cls._instance is None:
            cls._instance = super(VideoServer, cls).__new__(cls)
            cls._instance._init_server()
        return cls._instance

    def _init_server(self):
        """Initialize the Video server and configuration"""
        self._llm_config = get_llm_config_from_os_environ(
            llm_model_name="gpt-4o", server_name="Video Server"
        )

        # Initialize prompt templates
        self._video_analyze = (
            "input is a sequence of video frames. given user's task: {task}, "
            "analyze the video content following these steps:\n"
            "1. temporal sequence understanding\n"
            "2. motion and action analysis\n"
            "3. scene context interpretation\n"
            "4. object and person tracking\n"
            "return a json string with the following format: "
            '{{"video_analysis_result": "analysis result given task and video frames"}}'
        )

        self._video_extract_subtitles = (
            "input is a sequence of video frames. "
            "extract all subtitles (if present) in the video. "
            "return a json string with the following format: "
            '{{"video_subtitles": "extracted subtitles from video"}}'
        )

        self._video_summarize = (
            "input is a sequence of video frames. "
            "summarize the main content of the video. "
            "include key points, main topics, and important visual elements. "
            "return a json string with the following format: "
            '{{"video_summary": "concise summary of the video content"}}'
        )

        logger.info("VideoServer initialized")

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of VideoServer"""
        if cls._instance is None:
            return cls()
        return cls._instance

    @staticmethod
    def handle_error(e: Exception, operation_type: str) -> str:
        """Unified error handling and return standard format error message"""
        error_msg = f"{operation_type} error: {str(e)}"
        logger.error(f"{operation_type} operation failed: {str(e)}")
        logger.error(traceback.format_exc())
        return f'{{"error": "{error_msg}"}}'

    @staticmethod
    def get_video_frames(
        video_source: str, sample_rate: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Get frames from video with given sample rate using robust file handling

        Args:
            video_source: Path or URL to the video file
            sample_rate: Number of frames to sample per second

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing frame data and timestamp

        Raises:
            ValueError: When video file cannot be opened or is not a valid video
        """
        try:
            # Get file with validation (only video files allowed)
            file_path, _, _ = get_file_from_source(
                video_source,
                allowed_mime_prefixes=["video/"],
                max_size_mb=100.0,  # 100MB limit for videos
            )

            # Open video file
            video = cv2.VideoCapture(file_path)
            if not video.isOpened():
                raise ValueError(f"Could not open video file: {file_path}")

            fps = video.get(cv2.CAP_PROP_FPS)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            all_frames = []
            frames = []

            # Calculate frame interval based on sample rate
            frame_interval = max(1, int(fps / sample_rate))

            for i in range(0, frame_count):
                ret, frame = video.read()
                if not ret:
                    break

                # Convert frame to JPEG format
                _, buffer = cv2.imencode(".jpg", frame)
                frame_data = base64.b64encode(buffer).decode("utf-8")

                # Add data URL prefix for JPEG image
                frame_data = f"data:image/jpeg;base64,{frame_data}"

                all_frames.append({"data": frame_data, "time": i / fps})

            for i in range(0, len(all_frames), frame_interval):
                frames.append(all_frames[i])

            video.release()

            # Clean up temporary file if it was created for a URL
            if file_path != os.path.abspath(video_source) and os.path.exists(file_path):
                os.unlink(file_path)

            if not frames:
                raise ValueError(
                    f"Could not extract any frames from video: {video_source}"
                )

            return frames

        except Exception as e:
            logger.error(f"Error extracting frames from {video_source}: {str(e)}")
            raise

    @staticmethod
    def create_video_content(
        prompt: str, video_frames: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Create uniform video format for querying llm."""
        content = [{"type": "text", "text": prompt}]
        content.extend(
            [
                {"type": "image_url", "image_url": {"url": frame["data"]}}
                for frame in video_frames
            ]
        )
        return content

    @mcp
    @classmethod
    def analyze_video(
        cls,
        video_url: str = Field(description="The input video in given filepath or url."),
        question: str = Field(description="The question to analyze."),
        sample_rate: int = Field(default=2, description="Sample n frames per second."),
    ) -> str:
        """
        Analyze the video content by the given question.

        Args:
            video_url: Path or URL to the video file
            question: The question to analyze about the video
            sample_rate: Number of frames to sample per second

        Returns:
            JSON string containing analysis results
        """
        try:
            # Handle Field objects if they're passed directly
            if hasattr(video_url, "default") and not isinstance(video_url, str):
                video_url = video_url.default

            if hasattr(question, "default") and not isinstance(question, str):
                question = question.default

            if hasattr(sample_rate, "default") and not isinstance(sample_rate, int):
                sample_rate = sample_rate.default

            # Get the singleton instance
            instance = cls.get_instance()
            llm = get_llm_model(instance._llm_config)

            inputs = []
            video_frames = cls.get_video_frames(video_url, sample_rate)

            interval = 20
            frame_nums = 30
            all_res = []
            for i in range(0, len(video_frames), interval):
                inputs = []
                cur_frames = video_frames[i : min(i + frame_nums, len(video_frames))]
                content = cls.create_video_content(
                    instance._video_analyze.format(task=question), cur_frames
                )
                inputs.append({"role": "user", "content": content})
                try:
                    response = llm.completion(
                        messages=inputs,
                        temperature=0,
                    )
                    video_analysis_result = handle_llm_response(
                        response.content, "video_analysis_result"
                    )
                except Exception as e:
                    video_analysis_result = ""
                    cls.handle_error(e, "Error extracting frames")
                all_res.append(video_analysis_result)

            video_analysis_result = "\n".join(all_res)
            logger.info(f"Video analysis result: {video_analysis_result[:100]}...")
            return f'{{"video_analysis_result": "{video_analysis_result}"}}'

        except Exception as e:
            return cls.handle_error(e, "Video Analysis")

    # @mcp
    @classmethod
    def extract_video_subtitles(
        cls,
        video_url: str = Field(description="The input video in given filepath or url."),
        sample_rate: int = Field(default=2, description="Sample n frames per second."),
    ) -> str:
        """
        Extract subtitles from video frames if present.

        Args:
            video_url: Path or URL to the video file
            sample_rate: Number of frames to sample per second

        Returns:
            JSON string containing extracted subtitles
        """
        try:
            # Handle Field objects if they're passed directly
            if hasattr(video_url, "default") and not isinstance(video_url, str):
                video_url = video_url.default

            if hasattr(sample_rate, "default") and not isinstance(sample_rate, int):
                sample_rate = sample_rate.default

            # Get the singleton instance
            instance = cls.get_instance()
            llm = get_llm_model(instance._llm_config)

            inputs = []
            video_frames = cls.get_video_frames(video_url, sample_rate)
            content = cls.create_video_content(
                instance._video_extract_subtitles, video_frames
            )
            inputs.append({"role": "user", "content": content})

            response = llm.completion(
                messages=inputs,
                temperature=0,
            )
            video_subtitles = handle_llm_response(response.content, "video_subtitles")

            logger.info(f"Video subtitles extracted: {video_subtitles[:100]}...")
            return f'{{"video_subtitles": "{video_subtitles}"}}'

        except Exception as e:
            return cls.handle_error(e, "Video Subtitle Extraction")

    # @mcp
    @classmethod
    def summarize_video(
        cls,
        video_url: str = Field(description="The input video in given filepath or url."),
        sample_rate: int = Field(default=2, description="Sample n frames per second."),
    ) -> str:
        """
        Summarize the main content of the video.

        Args:
            video_url: Path or URL to the video file
            sample_rate: Number of frames to sample per second

        Returns:
            JSON string containing video summary
        """
        try:
            # Handle Field objects if they're passed directly
            if hasattr(video_url, "default") and not isinstance(video_url, str):
                video_url = video_url.default

            if hasattr(sample_rate, "default") and not isinstance(sample_rate, int):
                sample_rate = sample_rate.default

            # Get the singleton instance
            instance = cls.get_instance()
            llm = get_llm_model(instance._llm_config)

            inputs = []
            video_frames = cls.get_video_frames(video_url, sample_rate)
            content = cls.create_video_content(instance._video_summarize, video_frames)
            inputs.append({"role": "user", "content": content})

            response = llm.completion(
                messages=inputs,
                temperature=0,
            )
            video_summary = handle_llm_response(response.content, "video_summary")

            logger.info(f"Video summary generated: {video_summary[:100]}...")
            return f'{{"video_summary": "{video_summary}"}}'

        except Exception as e:
            return cls.handle_error(e, "Video Summarization")


if __name__ == "__main__":
    port = parse_port()

    video_server = VideoServer.get_instance()
    logger.info("VideoServer initialized and ready to handle requests")

    run_mcp_server(
        "Video Server",
        funcs=[
            video_server.analyze_video,
            video_server.extract_video_subtitles,
            video_server.summarize_video,
        ],
        port=port,
    )
