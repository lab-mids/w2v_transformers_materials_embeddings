#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor


class MultiObjectiveParetoAnalyzer:
    def __init__(self, objectives):
        """
        objectives: list of column names to optimize (must exist in each PKL DataFrame)

        This implementation assumes EXACTLY TWO objectives and will always compute
        two Pareto fronts:
        - [max, min]
        - [min, max]
        """
        if len(objectives) != 2:
            raise ValueError(
                f"This script assumes exactly 2 objectives for [max, min] / [min, max]. "
                f"Got {len(objectives)} objectives: {objectives!r}"
            )
        self.objectives = objectives

    def calculate_pareto_front(self, dataframe, directions, df_key=""):
        """
        dataframe: pandas DataFrame with the objective columns
        directions: list of 'max'/'min' (same length as self.objectives)
        df_key:    identifier (e.g. filename stem) used only for error messages
        """
        if len(directions) != len(self.objectives):
            raise ValueError(
                f"Length of directions ({len(directions)}) does not match "
                f"number of objectives ({len(self.objectives)}) for {df_key!r}."
            )

        # ensure objective columns exist
        missing = [c for c in self.objectives if c not in dataframe.columns]
        if missing:
            raise ValueError(f"Missing objective columns in {df_key}: {missing}")

        scores = dataframe[self.objectives].to_numpy(dtype=float)
        n = scores.shape[0]
        pareto = np.ones(n, dtype=bool)

        # pairwise dominance checks
        for i in range(n):
            if not pareto[i]:
                continue
            for j in range(i + 1, n):
                if not pareto[j]:
                    continue
                if self.is_dominated(scores[i], scores[j], directions):
                    pareto[i] = False
                    break
                elif self.is_dominated(scores[j], scores[i], directions):
                    pareto[j] = False
        return dataframe[pareto]

    def is_dominated(self, x, y, directions):
        """
        Return True if point x is dominated by point y under the given directions.
        'max' objective: larger is better.  'min' objective: smaller is better.
        """
        all_ge = True  # y is at least as good in all objectives
        any_gt = False  # y is strictly better in at least one objective

        for k, direction in enumerate(directions):
            if direction == 'max':
                if y[k] < x[k]:
                    all_ge = False
                    break
                if y[k] > x[k]:
                    any_gt = True
            elif direction == 'min':
                if y[k] > x[k]:
                    all_ge = False
                    break
                if y[k] < x[k]:
                    any_gt = True
            else:
                raise ValueError(f"Direction must be 'max' or 'min', got {direction!r}")
        return all_ge and any_gt

    def process_file(self, file_path, output_directory, output_format):
        """
        file_path: path to input PKL
        output_format: 'pkl' or 'csv'

        This will compute two Pareto fronts:
        - objectives directions = [max, min]
        - objectives directions = [min, max]

        Both are written into the SAME output file with a 'direction_mode' column
        indicating "max_min" or "min_max".
        """
        df_key = os.path.basename(file_path).rsplit('.', 1)[0]
        dataframe = pd.read_pickle(file_path)

        # directions for the two modes
        directions_max_min = ['max', 'min']
        directions_min_max = ['min', 'max']

        pareto_max_min_df = self.calculate_pareto_front(
            dataframe, directions_max_min, df_key=df_key
        ).copy()
        pareto_max_min_df["direction_mode"] = "max_min"

        pareto_min_max_df = self.calculate_pareto_front(
            dataframe, directions_min_max, df_key=df_key
        ).copy()
        pareto_min_max_df["direction_mode"] = "min_max"

        # Combine both fronts into a single DataFrame
        pareto_front_df = pd.concat([pareto_max_min_df, pareto_min_max_df], ignore_index=True)

        suffix = "_pareto_front." + ("pkl" if output_format == "pkl" else "csv")
        output_path = os.path.join(output_directory, df_key + suffix)

        os.makedirs(output_directory, exist_ok=True)
        if output_format == "pkl":
            pareto_front_df.to_pickle(output_path)
        else:
            pareto_front_df.to_csv(output_path, index=False)


def process_all_files_in_directory(input_directory, output_directory, objectives,
                                   num_workers=4, filename_suffix="_with_matscibert.pkl",
                                   output_format="pkl"):
    """
    Scan input_directory for files ending with filename_suffix, read PKL DataFrames,
    compute Pareto fronts ([max, min] and [min, max]), and write combined results
    to output_directory.
    """
    os.makedirs(output_directory, exist_ok=True)

    filenames = [
        f for f in os.listdir(input_directory)
        if f.endswith(filename_suffix) and os.path.isfile(os.path.join(input_directory, f))
    ]
    if not filenames:
        print(f"No files ending with '{filename_suffix}' found in {input_directory}")
        return

    analyzer = MultiObjectiveParetoAnalyzer(objectives)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                analyzer.process_file,
                os.path.join(input_directory, filename),
                output_directory,
                output_format
            )
            for filename in filenames
        ]
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f'Error processing file: {e}')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate Pareto fronts ([max, min] & [min, max]) for PKL datasets"
    )
    parser.add_argument("--input_directory", type=str, required=True, help="Input directory path")
    parser.add_argument("--output_directory", type=str, required=True, help="Output directory path")
    parser.add_argument(
        "--objectives",
        type=str,
        required=True,
        help='JSON list of TWO objective column names, e.g. ["Similarity_to_ORR","Some_Score"]'
    )
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument(
        "--filename_suffix",
        type=str,
        default="_with_matscibert.pkl",
        help="Only process files ending with this suffix"
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="csv",
        choices=["pkl", "csv"],
        help="Output format for Pareto front"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    objectives = json.loads(args.objectives)

    process_all_files_in_directory(
        args.input_directory,
        args.output_directory,
        objectives,
        num_workers=args.num_workers,
        filename_suffix=args.filename_suffix,
        output_format=args.output_format,
    )


if __name__ == "__main__":
    main()