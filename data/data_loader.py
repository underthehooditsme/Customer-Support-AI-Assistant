import os
import pandas as pd
import logging
import csv
from typing import Tuple
from datasets import load_dataset

from config import CONFIG

logger = logging.getLogger(__name__)

def download_dataset() -> pd.DataFrame:
    """
    Download dataset from Hugging Face.

    Returns:
        DataFrame containing the dataset
    """
    logger.info("Downloading dataset from Hugging Face")
    try:
        dataset_name = CONFIG.DATASET_NAME
        dataset = load_dataset(dataset_name)
        df = pd.DataFrame(dataset["train"])
        logger.info(f"Successfully downloaded dataset with {len(df)} entries")
        return df
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset to clean text and combine input-output pairs.

    Args:
        df: Raw dataset DataFrame with 'input' and 'output' columns

    Returns:
        Processed DataFrame
    """
    logger.info("Preprocessing dataset")

    df = df[['input', 'output']].dropna()
 
    df['input_clean'] = df['input'].str.replace(r'http\S+', '', regex=True)
    df['input_clean'] = df['input_clean'].str.replace(r'@\S+', '', regex=True)
    df['output_clean'] = df['output'].str.replace(r'http\S+', '', regex=True)
    df['output_clean'] = df['output_clean'].str.replace(r'@\S+', '', regex=True)

    # Combined text for embedding 
    df['combined_text'] = "Query: " + df['input_clean'] + " Response: " + df['output_clean']

    logger.info(f"Processed dataset with {len(df)} cleaned entries")
    return df

def load_or_create_processed_data() -> pd.DataFrame:
    """
    Load the processed dataset if it exists, or create it if it doesn't.

    Returns:
        Processed DataFrame
    """
    os.makedirs(CONFIG.DATA_DIR, exist_ok=True)

    if os.path.exists(CONFIG.PROCESSED_DATA_PATH):
        logger.info(f"Loading processed data from {CONFIG.PROCESSED_DATA_PATH}")
        return pd.read_csv(CONFIG.PROCESSED_DATA_PATH)
    else:
        logger.info("Processed data not found. Creating from raw dataset.")
        raw_df = download_dataset()
        processed_df = preprocess_dataset(raw_df)

        processed_df.to_csv(CONFIG.PROCESSED_DATA_PATH, index=False, quoting=csv.QUOTE_NONNUMERIC)
        logger.info(f"Saved processed data to {CONFIG.PROCESSED_DATA_PATH}")
        return processed_df

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    processed_data = load_or_create_processed_data()
    print(f"Loaded {len(processed_data)} processed samples")
    print(processed_data.head())
