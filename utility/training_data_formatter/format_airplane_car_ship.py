"""
@author: Adityam Ghosh
Date: 10-31-2023

"""

import pandas as pd
import os
import argparse

from collections import defaultdict
from rich.progress import track
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        "-D",
        required=True,
        type=str,
        help="the place where all the data are present",
    )

    parser.add_argument(
        "--save_dir", "-s", required=True, type=str, help="where to save the data"
    )

    parser.add_argument(
        "--val_size",
        "-v",
        required=False,
        type=float,
        default=0.02,
        help="the validation test size",
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    save_dir = args.save_dir
    val_size = args.val_size

    data = defaultdict(lambda: list())

    for root, folder, files in os.walk(data_dir):
        n_files = len(files)
        for i in track(range(n_files)):
            file = files[i]
            label = root.split("/")[-1]

            data["image_path"].append(os.path.join(root, file))
            data["label"].append(label)

    data = pd.DataFrame(data)

    data["text"] = data["label"].apply(lambda x: f"This is an image of a {x}")
    data = data.sample(frac=1, random_state=32)
    print(data["label"].value_counts())

    train_data, val_data = train_test_split(
        data, test_size=val_size, random_state=32, shuffle=True, stratify=data["label"]
    )

    data["data_use_for"] = "train"
    data.loc[val_data.index, "data_use_for"] = "validation"

    data.to_csv(save_dir + "/airplane_car_ship.csv", index=False)
