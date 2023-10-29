"""
@author: Adityam Ghosh
Date: 10-30-2023

"""

import numpy as np
import pandas as pd
import os
import argparse
import json

from collections import defaultdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="format_animals_151",
        description="data script for formatting animals_151 dataset",
    )

    parser.add_argument(
        "--data_dir",
        "-D",
        required=True,
        type=str,
        help="the directory where the data is stored",
    )

    args = parser.parse_args()

    labels_file = ""
    data = defaultdict(lambda: list())

    for root, folders, files in os.walk(args.data_dir):
        for file in files:
            if ".csv" in file:
                continue

            if ".json" in file:
                labels_file = os.path.join(root, file)
            else:
                sci_label = root.split("/")[-1]
                img_file = os.path.join(root, file)

                data["label"].append(sci_label)
                data["image_path"].append(img_file)

    mapping_label = dict()

    with open(labels_file, "r") as fp:
        mapping_label = json.load(fp)

    df = pd.DataFrame(data)
    df["label"] = df["label"].map(mapping_label)
    df["text"] = df["label"].apply(lambda x: f"A photo of a {x.lower()}")

    df = df.sample(frac=1, random_state=32)

    df.to_csv(os.path.join(args.data_dir, "animals_151_csv.csv"), index=False)
