# coding: utf-8
# Copyright (c) 2025 inclusionAI.
import os

from multipart import file_path


def mock_dataset(name: str):
    if name == 'gaia':
        current_work_dir = os.path.abspath(os.path.dirname(__file__))
        # file_path = os.path.join(current_work_dir, 'gaia/32102e3e-d12a-4209-9163-7b3a104efe5d.xlsx')
        file_path = "/Users/again/Downloads/abc.xlsx"
        return f'The attached spreadsheet shows the inventory for a movie and video game rental store in Seattle, Washington. What is the title of the oldest Blu-Ray recorded in this spreadsheet? Return it as appearing in the spreadsheet. Here are the necessary table files: {file_path}, for processing excel file, you can write python code and leverage excel toolkit to process the file step-by-step and get the information.'
    return None