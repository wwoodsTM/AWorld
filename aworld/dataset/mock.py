# coding: utf-8
# Copyright (c) 2025 inclusionAI.
import os

import pandas as pd
import numpy as np


def mock_dataset(name: str):
    if name == 'gaia':
        current_work_dir = os.path.abspath(os.path.dirname(__file__))
        file_path = os.path.join(current_work_dir, 'gaia/gaia.npy')

        numpy_array = np.load(file_path, allow_pickle=True)
        df = pd.DataFrame(numpy_array[:-1])
        query = numpy_array[-1][0]

        save_file_path = os.path.join(current_work_dir, 'gaia/gaia.xlsx')
        df.to_excel(save_file_path, index=False, header=None)
        return query.format(file_path=save_file_path)
    return None
