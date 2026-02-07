import argparse
from matnexus import TextProcessor
import pandas as pd
import nltk


def parse_args():
    parser = argparse.ArgumentParser(description="Process collected papers")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Input path for the collected papers CSV file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for the processed papers CSV file",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    nltk.download("stopwords")

    data = pd.read_csv(args.input_path)
    text_processor = TextProcessor.TextProcessor(data)
    processed_df = text_processor.processed_df
    processed_df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
