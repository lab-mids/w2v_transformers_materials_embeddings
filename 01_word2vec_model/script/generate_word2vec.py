import argparse
from matnexus import VecGenerator
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Generate word2vec model")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Input path for the processed papers CSV file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Output path for the generated word2vec model",
    )
    parser.add_argument(
        "--sg",
        type=int,
        default=1,
        help="Training algorithm: 1 for skip-gram, 0 for CBOW",
    )
    parser.add_argument(
        "--vector_size",
        type=int,
        default=200,
        help="Dimensionality of the word vectors",
    )
    parser.add_argument(
        "--hs",
        type=int,
        default=1,
        help="If 1, hierarchical softmax will be used for model training",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Maximum distance between the current and predicted word",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=1,
        help="Ignores all words with total frequency lower than this",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads to train the model",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    processed_df = pd.read_csv(args.input_path)
    corpus = VecGenerator.Corpus(processed_df)
    sentences = corpus.sentences

    vec_generator = VecGenerator.Word2VecModel(sentences)
    vec_generator.fit(
        sg=args.sg,
        vector_size=args.vector_size,
        hs=args.hs,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
    )
    vec_generator.save(args.model_path)


if __name__ == "__main__":
    main()
