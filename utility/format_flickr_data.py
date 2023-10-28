"""
@author: Adityam Ghosh
Date: 10-29-2023

"""

import argparse
import os
import pandas as pd

from collections import defaultdict
from sklearn.model_selection import train_test_split
from copy import deepcopy

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="flickr_formatter", description="Formats the flickr data into a csv path")

    parser.add_argument("--data_dir", "-d", required=True, type=str,
                        help="the directory which contains the flickr data")
    parser.add_argument("--test_size", "-s", required=False,
                        type=float, default=.2, help="the size of the test dataset")

    args = parser.parse_args()

    _data = defaultdict(lambda: list())

    for root, folders, files in os.walk(args.data_dir):
        for file in files:
            path = os.path.join(root, file)
            if ".txt" in file:
                _data["captions_path"].append(path)
            else:
                _data["img_path"].append(path)

    captions = pd.read_table(_data["captions_path"][0], sep=",")
    captions.columns = ["image_id", "caption"]
    del _data["captions_path"]

    imgs = pd.DataFrame(_data["img_path"], columns=["img_path"])
    imgs["image_id"] = imgs["img_path"].apply(lambda x: x.split("/")[-1])

    merged_df = pd.merge(captions, imgs, how="inner", on="image_id")

    train_data, dummy_data = train_test_split(
        merged_df, test_size=args.test_size, random_state=32, shuffle=True)

    val_data, test_data = train_test_split(
        dummy_data, test_size=.5, shuffle=False)

    final_merged_df = deepcopy(merged_df)
    final_merged_df["data_use_for"] = "train"
    final_merged_df.loc[val_data.index, "data_use_for"] = "validation"
    final_merged_df.loc[test_data.index, "data_use_for"] = "test"

    final_merged_df.columns = ["image_id",
                               "text", "image_path", "data_use_for"]
    final_merged_df.to_csv(args.data_dir+"/flickr8k_csv.csv", index=False)
