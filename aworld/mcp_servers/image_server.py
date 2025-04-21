"""
Image MCP Server

This module provides MCP server functionality for image processing and analysis.
It handles image encoding, optimization, and various image analysis tasks such as
OCR (Optical Character Recognition) and visual reasoning.

The server supports both local image files and remote image URLs with proper validation
and handles various image formats including JPEG, PNG, GIF, and others.

Main functions:
- encode_images: Encodes images to base64 format with optimization
- optimize_image: Resizes and optimizes images for better performance
- Various MCP tools for image analysis and processing
"""

import base64
import os
import traceback
from collections import Counter
from io import BytesIO
from typing import Any, Dict, List

from PIL import Image
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
from aworld.models.llm import get_llm_model, call_llm_model


class ImageServer(MCPServerBase):
    """
    Image Server class for image processing and analysis.

    This class provides methods for image encoding, optimization, and various
    image analysis tasks such as OCR and visual reasoning.
    """

    _instance = None
    _name = "image"
    _llm_config = None
    _image_ocr_prompt = None
    _image_reasoning_prompt = None

    def __new__(cls):
        """Implement singleton pattern"""
        if cls._instance is None:
            cls._instance = super(ImageServer, cls).__new__(cls)
            cls._instance._init_server()
        return cls._instance

    def _init_server(self):
        """Initialize the Image server and configuration"""
        self._llm_config = get_llm_config_from_os_environ(
            llm_model_name="gpt-4o", server_name="Image Server"
        )

        self._image_ocr_prompt = (
            "Input is a base64 encoded image. Read text from image if present. "
            "Return a json string with the following format: "
            '{{"image_text": "text from image"}}'
        )

        self._image_reasoning_prompt = (
            "Input is a base64 encoded image. Given user's task: {task}, "
            "solve it following the guide line:\n"
            "1. Careful visual inspection\n"
            "2. Contextual reasoning\n"
            "3. Text transcription where relevant\n"
            "4. Logical deduction from visual evidence\n"
            "Return a json string with the following format: "
            '{{"image_reasoning_result": "reasoning result given task and image"}}'
        )

        logger.info("ImageServer initialized")

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of ImageServer"""
        if cls._instance is None:
            return cls()
        return cls._instance

    @staticmethod
    def optimize_image(image_data: bytes, max_size: int = 1024) -> bytes:
        """
        Optimize image by resizing if needed

        Args:
            image_data: Raw image data
            max_size: Maximum dimension size in pixels

        Returns:
            bytes: Optimized image data

        Raises:
            ValueError: When image cannot be processed
        """
        try:
            image = Image.open(BytesIO(image_data))

            # Resize if image is too large
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            # Save to buffer
            buffered = BytesIO()
            image_format = image.format if image.format else "JPEG"
            image.save(buffered, format=image_format)
            return buffered.getvalue()

        except Exception as e:
            logger.warning(f"Failed to optimize image: {str(e)}")
            return image_data  # Return original data if optimization fails

    @classmethod
    def encode_images(
        cls, image_sources: List[str], with_header: bool = True
    ) -> List[str]:
        """
        Encode images to base64 format with robust file handling

        Args:
            image_sources: List of URLs or local file paths of images
            with_header: Whether to include MIME type header

        Returns:
            List[str]: Base64 encoded image strings, with MIME type prefix if with_header is True

        Raises:
            ValueError: When image source is invalid or image format is not supported
        """
        if not image_sources:
            raise ValueError("Image sources cannot be empty")

        images = []
        for image_source in image_sources:
            try:
                # Get file with validation (only image files allowed)
                file_path, mime_type, content = get_file_from_source(
                    image_source,
                    allowed_mime_prefixes=["image/"],
                    max_size_mb=10.0,  # 10MB limit for images
                )

                # Optimize image
                optimized_content = cls.optimize_image(content)

                # Encode to base64
                image_base64 = base64.b64encode(optimized_content).decode()

                # Format with header if requested
                final_image = (
                    f"data:{mime_type};base64,{image_base64}"
                    if with_header
                    else image_base64
                )

                images.append(final_image)

                # Clean up temporary file if it was created for a URL
                if file_path != os.path.abspath(image_source) and os.path.exists(
                    file_path
                ):
                    os.unlink(file_path)

            except Exception as e:
                logger.error(f"Error encoding image from {image_source}: {str(e)}")
                raise

        return images

    @staticmethod
    def create_image_contents(
        prompt: str, image_base64: List[str]
    ) -> List[Dict[str, Any]]:
        """Create uniform image format for querying llm."""
        content = [
            {"type": "text", "text": prompt},
        ]
        content.extend(
            [{"type": "image_url", "image_url": {"url": url}} for url in image_base64]
        )
        return content

    @classmethod
    def ocr_image(
        cls,
        image_urls: List[str] = Field(
            description="The input image in given a list of filepaths or urls."
        ),
    ) -> str:
        """Read text (if present) from the given image in given a list of filepaths or urls."""
        instance = cls.get_instance()
        llm = get_llm_model(instance._llm_config)

        inputs = []
        try:
            image_base64 = cls.encode_images(image_urls)
            content = cls.create_image_contents(
                instance._image_ocr_prompt, image_base64
            )
            inputs.append({"role": "user", "content": content})

            response = llm.completion(
                messages=inputs,
                temperature=0,
            )
            image_text = handle_llm_response(response.content, "image_text")
        except (ValueError, IOError, RuntimeError) as e:
            logger.error(f"image_text-Execute error: {traceback.format_exc()}")
            image_text = ""

        logger.info(f"---get_text_by_ocr-image_text:{image_text}")

        return image_text

    @mcp
    @classmethod
    def reasoning_image(
        cls,
        image_urls: List[str] = Field(
            description="The input image(s) in given a list of filepaths or urls."
        ),
        question: str = Field(description="The question to ask."),
    ) -> str:
        """Solve the question by careful reasoning given the image(s) in given filepath or url."""
        instance = cls.get_instance()
        llm = get_llm_model(instance._llm_config)

        inputs = []
        try:
            image_base64 = cls.encode_images(image_urls)
            prompt = instance._image_reasoning_prompt.format(task=question)
            content = cls.create_image_contents(prompt, image_base64)
            inputs.append({"role": "user", "content": content})

            response = call_llm_model(
                llm,
                messages=inputs,
                temperature=0,
            )
            instance._token_usage = dict(Counter(response.usage) + Counter(instance._token_usage))
            reasoning_result = handle_llm_response(
                response.content, "image_reasoning_result"
            )
        except (ValueError, IOError, RuntimeError) as e:
            logger.error(f"image_reasoning-Execute error: {traceback.format_exc()}")
            reasoning_result = ""

        logger.info(f"---reasoning_image-reasoning_result:{reasoning_result}")

        return reasoning_result


# Main function
if __name__ == "__main__":
    port = parse_port()

    image_server = ImageServer.get_instance()
    logger.info("ImageServer initialized and ready to handle requests")

    run_mcp_server(
        "Image Server",
        funcs=[
            image_server.ocr_image,
            image_server.reasoning_image,
        ],
        port=port,
    )
