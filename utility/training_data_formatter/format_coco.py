"""
@author: Adityam Ghosh
Date: 11-03-2023
"""

import polars as pl
import json
import argparse
import os


def process_json_files(json_path: str, img_root_path: str) -> pl.DataFrame:
    """A function to export a dataframe based on json file"""
    with open(json_path, "r") as fp:
        data = json.load(fp)

    images = data["images"]
    annotations = data["annotations"]

    img_df = pl.from_dicts(images)
    anno_df = pl.from_dicts(annotations)

    combined_df = img_df.join(anno_df, left_on="id", right_on="image_id", how="inner")
    n_rows = combined_df.shape[0]
    combined_df = combined_df.with_columns(
        pl.Series(name="image_path", values=[img_root_path] * n_rows)
    )

    return combined_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_annotations",
        "-t",
        required=True,
        type=str,
        help="the path which contains the train caption json file",
    )

    parser.add_argument(
        "--val_annotations",
        "-v",
        required=True,
        type=str,
        help="the path which contains the validation caption json file",
    )

    parser.add_argument(
        "--train_img_path",
        "-T",
        required=True,
        type=str,
        help="root dir which contains the train_image.zip file",
    )

    parser.add_argument(
        "--val_img_path",
        "-V",
        required=True,
        type=str,
        help="root dir which contains the val_image.zip file",
    )

    parser.add_argument(
        "--save_loc",
        "-s",
        required=False,
        type=str,
        default=".",
        help="path on where to save the csv files",
    )

    args = parser.parse_args()

    train_anno_path = args.train_annotations
    val_anno_path = args.val_annotations
    train_img_path = args.train_img_path
    val_img_path = args.val_img_path
    save_loc = args.save_loc

    train_df = process_json_files(train_anno_path, train_img_path)
    val_df = process_json_files(val_anno_path, val_img_path)

    print(train_df)
    print(train_df.shape)

    train_df.write_csv(os.path.join(save_loc, "train.csv"), separator=",", batch_size=1)
    val_df.write_csv(os.path.join(save_loc, "val.csv"), separator=",", batch_size=1)
