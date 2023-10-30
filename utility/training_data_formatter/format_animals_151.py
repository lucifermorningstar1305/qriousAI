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
from sklearn.model_selection import train_test_split

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

    parser.add_argument(
        "--save_dir",
        "-s",
        required=True,
        type=str,
        help="the directory to save the final data",
    )
    parser.add_argument(
        "--val_size",
        "-v",
        required=False,
        type=float,
        help="the validation size of the data",
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    save_dir = args.save_dir
    val_sizze = args.val_size

    labels_file = ""
    data = defaultdict(lambda: list())

    for root, folders, files in os.walk(data_dir):
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
    df["text"] = df["label"].apply(
        lambda x: f"This is an image of an animal which is identified as {x.lower()}"
    )

    df = df.sample(frac=1, random_state=32)
    train_data, val_data = train_test_split(
        df, test_size=val_sizze, shuffle=True, random_state=32, stratify=df["label"]
    )

    df["data_use_for"] = "train"
    df.loc[val_data.index, "data_use_for"] = "validation"

    df.to_csv(os.path.join(save_dir, "animals_151_csv.csv"), index=False)
