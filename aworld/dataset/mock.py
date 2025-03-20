# coding: utf-8
# Copyright (c) 2025 inclusionAI.


def mock_dataset(name: str):
    """Mock data, need modify `/Users/your_name/Downloads/example.xlsx` to the real file path."""
    if name == 'gaia':
        return 'The attached spreadsheet shows the inventory for a movie and video game rental store in Seattle, Washington. What is the title of the oldest Blu-Ray recorded in this spreadsheet? Return it as appearing in the spreadsheet. Here are the necessary table files: /Users/yourname/Downloads/example.xlsx, for processing excel file, you can write python code and leverage excel toolkit to process the file step-by-step and get the information.'
    return None