from aworld import logger
from aworld.config import ToolConfig
from aworld.virtual_environments.document.document import DocumentTool
import traceback


class TestDocumentTool():
    def __init__(self):
        conf = ToolConfig(use_vision=True)
        self.document_tool = DocumentTool(conf=conf)

    def test_document_analysis(self,document_path):
        try:
            content, keyframes, error = self.document_tool.document_analysis(document_path)
            # content, keyframes, error
            logger.info(f"document_path:{document_path}")
            logger.info(f"keyframes:{keyframes}")
        except Exception as e:
            logger.error(f"error: {e}")
            traceback.print_exc()


if __name__ == '__main__':
    document_path = "/path/video.mp4"
    TestDocumentTool().test_document_analysis(document_path)