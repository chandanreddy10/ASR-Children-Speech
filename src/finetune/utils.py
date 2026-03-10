import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def split_dataframe(df, output_dir, n_splits=3):

    # create path relative to project root
    output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # split dataframe
    splits = np.array_split(df, n_splits)

    for i, split_df in enumerate(splits):
        file_path = output_dir / f"split_{i}.csv"
        split_df.to_csv(file_path, index=False)

    return splits