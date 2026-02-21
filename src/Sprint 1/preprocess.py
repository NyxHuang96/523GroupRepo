"""
File: preprocess.py
Author: Tianhao Cao
Date: 2026-02-20
Last Updated: 2026-02-21
Course: COLX 523
Description: Download data from kaggle through official APi, preprocess the Spam/Ham/Phish email dataset, and export to data/processed/corpus.json
"""

import pandas as pd
import os
import re
from pathlib import Path
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter


def preprocess_dataset():
    script_dir = Path(__file__).resolve().parent

    processed_dir = script_dir.parent.parent / "data" / "processed"
    os.makedirs(processed_dir, exist_ok=True)
    processed_data_path = processed_dir / "corpus.json"

    # Step 1: Define file paths
    # kagglehub documentation: https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
    # Set the path to the file you'd like to load
    file_path = "df.csv"

    # Load the latest version
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "akshatsharma2/the-biggest-spam-ham-phish-email-dataset-300000",
        file_path,
        # Provide any additional arguments like
        # sql_query or pandas_kwargs. See the
        # documenation for more information:
        # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
    )

    print("-" * 50)
    print("First 5 records:", df.head())

    # Initial Inspection
    print("-" * 50)
    print(f"Original Dataset Shape: {df.shape}")
    print("First 5 records:")
    print(df.head())

    # Handle Missing Values
    initial_count = len(df)
    df = df.dropna(subset=["text", "label"])
    missing_removed = initial_count - len(df)
    print("-" * 50)
    print(f"Removed {missing_removed} rows with missing values.")

    # Handle Duplicates
    initial_count = len(df)
    df = df.drop_duplicates(subset=["text"])
    duplicates_removed = initial_count - len(df)
    print(f"Removed {duplicates_removed} duplicate rows.")

    # Map the numeric labels to text categories
    # According to resources.md: 0 -> Ham, 1 -> Phish, 2 -> Spam
    label_mapping = {0: "Ham", 1: "Phish", 2: "Spam"}
    df["label"] = df["label"].map(label_mapping)
    print("-" * 50)
    print(f"Label Distribution after mapping:\n{df['label'].value_counts()}")

    # Basic Text Cleaning
    print("-" * 50)
    print("Cleaning text data (lowercase, removing extra whitespaces)...")

    df["text"] = df["text"].str.lower()
    df["text"] = df["text"].apply(lambda x: re.sub(r"\s+", " ", str(x)).strip())

    # Save the Processed Dataset
    print("-" * 50)
    print(f"Saving processed data to: {processed_data_path}")

    df.to_json(processed_data_path, orient="records", lines=True)
    print(f"Final Dataset Shape: {df.shape}")
    print("-" * 50)


if __name__ == "__main__":
    preprocess_dataset()
