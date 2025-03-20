import json
import os
import base64
import tempfile
import subprocess
from typing import Any, Dict, Tuple
from urllib.parse import urlparse

import xmltodict
from pydantic import BaseModel

from aworld.agents.browser.utils import encode_image_from_file, encode_image_from_url
from aworld.config import ToolConfig
from aworld.core.env.tool_action import DocumentExecuteAction
from aworld.core.common import Tools, Observation, ActionModel, ActionResult
from aworld.core.env.env_tool import ToolFactory, EnvTool
from aworld.logs.util import logger


class InputDocument(BaseModel):
    document_path: str | None = None


@ToolFactory.register(name=Tools.DOCUMENT_ANALYSIS.value, desc="document analysis",
                      supported_action=DocumentExecuteAction)
class DocumentTool(EnvTool[Observation, ActionModel]):
    def __init__(self, conf: ToolConfig, **kwargs) -> None:
        """Init document tool."""
        super(DocumentTool, self).__init__(conf, **kwargs)
        self._observation_space = self.observation_space()
        self._action_space = self.action_space()
        self.cur_observation = None
        self.content = None
        self.keyframes = []
        self.init()
        self.step_finished = True

    def observation_space(self):
        pass

    def action_space(self):
        pass

    def reset(self, *, seed: int | None = None, options: Dict[str, str] | None = None) -> Tuple[
        Observation, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        self.close()
        observation = self._get_observation()
        observation.action_result = [ActionResult(extracted_content='start', include_in_memory=True)]
        self.cur_observation = observation
        self.step_finished = True
        return observation, {}

    def init(self) -> None:
        self.initialized = True

    def finished(self) -> bool:
        return True

    def close(self) -> None:
        if hasattr(self, 'context') and self.context:
            self.context.close()

    def finished(self) -> bool:
        return self.step_finished

    def step(self, actions: list[ActionModel], **kwargs) -> Tuple[Observation, float, bool, bool, Dict[str, Any]]:
        self.step_finished = False
        reward = 0
        fail_error = ""
        observation: 'Observation' = Observation(**{
            'dom_tree': '',
            'image': '',
            'action_result': [],
            'info': {}
        })
        try:
            if not actions:
                raise ValueError("actions is empty")
            action = actions[0]
            document_path = action.params.get("document_path", "")
            if not document_path:
                raise ValueError("document is unvaild")
            output,keyframes,error = self.document_analysis(document_path)
            observation.action_result.append(
                ActionResult(is_done=True,
                             success=False if error else True,
                             content=f"{output}",
                             key_frame=f"{keyframes}",
                             error=f"{error}",
                             include_in_memory=False))
        except Exception as e:
            fail_error = str(e)
        finally:
            self.step_finished = True
        return (observation, reward, kwargs.get("terminated", False),
                kwargs.get("truncated", False), {
                    "exception": fail_error
                })

    def document_analysis(self, document_path):
        error = None
        try:
            if any(document_path.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
                parsed_url = urlparse(document_path)
                is_url = all([parsed_url.scheme, parsed_url.netloc])
                if not is_url:
                    base64_image = encode_image_from_file(document_path)
                else:
                    base64_image = encode_image_from_url(document_path)
                self.content = f"data:image/jpeg;base64,{base64_image}"

            if any(document_path.endswith(ext) for ext in ["xls", "xlsx"]):
                try:
                    try:
                        import pandas as pd
                    except ImportError:
                        error = "pandas library not found. Please install pandas: pip install pandas"
                        return self.content, self.keyframes, error
                    
                    excel_data = {}
                    
                    with pd.ExcelFile(document_path) as xls:
                        sheet_names = xls.sheet_names
                        for sheet_name in sheet_names:
                            df = pd.read_excel(xls, sheet_name=sheet_name)
                            sheet_data = df.to_dict(orient='records')
                            excel_data[sheet_name] = sheet_data
                    
                    self.content = json.dumps(excel_data, ensure_ascii=False)
                    logger.info(f"Successfully processed Excel file: {document_path}")
                    logger.info(f"Found {len(sheet_names)} sheets: {', '.join(sheet_names)}")
                    
                except Exception as excel_error:
                    error = str(excel_error)

            if any(document_path.endswith(ext) for ext in ["json", "jsonl", "jsonld"]):
                with open(document_path, "r", encoding="utf-8") as f:
                    self.content = json.load(f)
                f.close()

            if any(document_path.endswith(ext) for ext in ["xml"]):
                data = None
                with open(document_path, "r", encoding="utf-8") as f:
                    data = f.read()
                f.close()

                try:
                    self.content = xmltodict.parse(data)
                    logger.info(f"The extracted xml data is: {self.content}")

                except Exception as e:
                    logger.info(f"The raw xml data is: {data}")
                    error = str(e)
                    self.content = data

            if any(document_path.endswith(ext) for ext in ["doc", "docx"]):
                from docx2markdown._docx_to_markdown import docx_to_markdown
                file_name = os.path.basename(document_path)
                md_file_path = f"{file_name}.md"
                docx_to_markdown(document_path, md_file_path)
                with open(md_file_path, "r") as f:
                    self.content = f.read()
                f.close()
            
            if any(document_path.endswith(ext) for ext in ["pdf"]):
                # try using pypdf to extract text from pdf
                try:
                    from PyPDF2 import PdfReader

                    # Open file in binary mode for PdfReader
                    f = open(document_path, "rb")
                    reader = PdfReader(f)
                    extracted_text = ""
                    for page in reader.pages:
                        extracted_text += page.extract_text()
                    self.content = extracted_text
                    f.close()
                except Exception as pdf_error:
                    error = str(pdf_error)
            
            # audio
            if any(document_path.endswith(ext.lower()) for ext in [".mp3", ".wav", ".wave"]):
                try:
                    # audio-> base64
                    with open(document_path, "rb") as audio_file:
                        audio_bytes = audio_file.read()
                        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                    
                    # ext
                    ext = os.path.splitext(document_path)[1].lower()
                    mime_type = "audio/mpeg" if ext == ".mp3" else "audio/wav"
                    
                    # data URI
                    self.content = f"data:{mime_type};base64,{audio_base64}"
                except Exception as audio_error:
                    error = str(audio_error)
                    logger.error(f"Error processing audio file: {error}")

            # video
            if any(document_path.endswith(ext.lower()) for ext in [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]):
                try:
                    try:
                        import cv2
                        import numpy as np
                    except ImportError:
                        error = "Required libraries not found. Please install opencv-python: pip install opencv-python"
                        return None,None,error
                    
                    # create temp dir
                    temp_dir = tempfile.mkdtemp()
                    
                    # 1.get audio -> base64
                    audio_path = os.path.join(temp_dir, "extracted_audio.mp3")
                    
                    #  get audio by ffmpeg
                    try:
                        subprocess.run([
                            "ffmpeg", "-i", document_path, "-q:a", "0", 
                            "-map", "a", audio_path, "-y"
                        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        
                        # audio->base64
                        with open(audio_path, "rb") as audio_file:
                            audio_bytes = audio_file.read()
                            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                        
                        audio_data_uri = f"data:audio/mpeg;base64,{audio_base64}"
                    except (subprocess.SubprocessError, FileNotFoundError) as e:
                        logger.warning(f"Failed to extract audio: {str(e)}")
                        audio_data_uri = None
                    
                    # 2. get keyframes
                    cap = cv2.VideoCapture(document_path)
                    
                    if not cap.isOpened():
                        raise ValueError(f"Could not open video file: {document_path}")
                    
                    # get video message
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps if fps > 0 else 0
                    
                    # keyframes policy- per duration/10s，max 10
                    keyframes_count = min(10, int(frame_count))
                    frames_interval = max(1, int(frame_count / keyframes_count))
                    
                    self.keyframes = []
                    frame_index = 0
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # per frames_interval save
                        if frame_index % frames_interval == 0:
                            # save JPEG -> base64
                            _, buffer = cv2.imencode(".jpg", frame)
                            img_base64 = base64.b64encode(buffer).decode('utf-8')
                            time_position = frame_index / fps if fps > 0 else 0
                            
                            self.keyframes.append(f"data:image/jpeg;base64,{img_base64}")
                            
                            if len(self.keyframes) >= keyframes_count:
                                break
                        
                        frame_index += 1
                    
                    cap.release()

                    self.content = audio_data_uri
                    logger.info(f"Successfully processed video file: {document_path}")
                    logger.info(f"Extracted {len(self.keyframes)} keyframes and audio track")
                    # clean tmp files
                    try:
                        os.remove(audio_path)
                        os.rmdir(temp_dir)
                    except Exception as cleanup_error:
                        logger.warning(f"Error cleaning up temp files: {str(cleanup_error)}")
                
                except Exception as video_error:
                    error = str(video_error)
                    logger.error(f"Error processing video file: {error}")

        finally:
            pass

        return self.content,self.keyframes,error
