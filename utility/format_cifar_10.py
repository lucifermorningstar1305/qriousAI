"""
@author: Adityam Ghosh
Date: 10-30-2023

"""

from typing import List, Dict, Any, Callable

import numpy as np
import pandas as pd
import argparse
import pickle
import os

from collections import defaultdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir", "-D", required=True, type=str, help="the directory of the images"
    )

    parser.add_argument(
        "--save_csv_loc",
        "-s",
        required=True,
        type=str,
        help="the directory on where to save the file",
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    save_path = args.save_csv_loc

    data = defaultdict(lambda: list())

    for root, folders, files in os.walk(data_dir):
        for file in files:
            img_path = os.path.join(root, file)
            label = root.split("/")[-1]

            data["image_path"].append(img_path)
            data["label"].append(label)

    data = pd.DataFrame(data)
    data["text"] = data["label"].apply(lambda x: f"A photo of a {x}")

    data = data.sample(frac=1, random_state=32)

    data.to_csv(save_path + "/cifar_csv.csv", index=False)
