"""
@author: Adityam Ghosh
Date: 10-31-2023

"""

import pandas as pd
import os
import argparse
import random
import gc

from collections import defaultdict
from rich.progress import track
from glob import glob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        "-d",
        required=True,
        type=str,
        help="the data directory which contains all csv files",
    )

    args = parser.parse_args()
    data_dir = args.data_dir

    csv_files = glob(os.path.join(data_dir, "*.csv"))

    dfs = []

    for csv_file in csv_files:
        if "final_df" in csv_file:
            continue

        _df = pd.read_csv(csv_file)
        dfs.append(_df)

    final_df = pd.concat(dfs, axis=0)
    final_df = final_df.sample(frac=1, random_state=32)
    print(final_df.shape)

    final_df.to_csv(os.path.join(data_dir, "final_df_csv.csv"), index=False)
