import argparse
import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import json


class MultiObjectiveParetoAnalyzer:
    def __init__(self, objectives):
        """
        objectives: list of column names (assumed length 2 for (max, min) and (min, max))
        """
        self.objectives = objectives
        if len(self.objectives) != 2:
            raise ValueError(
                f"Expected exactly 2 objectives for (max, min) and (min, max), "
                f"got {len(self.objectives)}: {self.objectives}"
            )

    def calculate_pareto_front(self, dataframe, directions):
        """
        directions: list/tuple of 'max' or 'min' for each objective, e.g. ['max', 'min']
        """
        if len(directions) != len(self.objectives):
            raise ValueError(
                f"Directions length {len(directions)} does not match "
                f"number of objectives {len(self.objectives)}"
            )

        scores = dataframe[self.objectives].to_numpy()
        population_size = scores.shape[0]
        pareto_front = np.ones(population_size, dtype=bool)

        for i in range(population_size):
            if not pareto_front[i]:
                continue
            for j in range(i + 1, population_size):
                if not pareto_front[j]:
                    continue
                if self.is_dominated(scores[i], scores[j], directions):
                    pareto_front[i] = False
                    break
                elif self.is_dominated(scores[j], scores[i], directions):
                    pareto_front[j] = False

        return dataframe[pareto_front]

    def is_dominated(self, x, y, directions):
        # True if x is dominated by y
        strictly_better = False

        for i, direction in enumerate(directions):
            if direction == "min":
                if y[i] > x[i]:  # y worse -> cannot dominate
                    return False
                if y[i] < x[i]:
                    strictly_better = True
            elif direction == "max":
                if y[i] < x[i]:  # y worse -> cannot dominate
                    return False
                if y[i] > x[i]:
                    strictly_better = True
            else:
                raise ValueError(f"Unknown direction '{direction}'")

        return strictly_better

    def process_file(self, file_path, output_directory):
        df_key = os.path.basename(file_path).rsplit('.', 1)[0]
        dataframe = pd.read_csv(file_path)

        # Two direction combinations: (max, min) and (min, max)
        directions_max_min = ['max', 'min']
        directions_min_max = ['min', 'max']

        pareto_max_min = self.calculate_pareto_front(dataframe, directions_max_min)
        pareto_min_max = self.calculate_pareto_front(dataframe, directions_min_max)

        # Union of both sets of candidates, drop exact duplicate rows
        combined_pareto = pd.concat([pareto_max_min, pareto_min_max], axis=0)
        combined_pareto = combined_pareto.drop_duplicates().reset_index(drop=True)

        output_filename = f'{df_key}_pareto_front.csv'
        output_path = os.path.join(output_directory, output_filename)

        combined_pareto.to_csv(output_path, index=False)


def process_all_files_in_directory(input_directory, output_directory, objectives,
                                   num_workers=4):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    filenames = [
        f for f in os.listdir(input_directory)
        if f.endswith('_material_system_with_similarity.csv')
    ]

    analyzer = MultiObjectiveParetoAnalyzer(objectives)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                analyzer.process_file,
                os.path.join(input_directory, filename),
                output_directory
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
        description="Calculate Pareto fronts for datasets with both (max, min) and (min, max) "
                    "and write a unified _pareto_front.csv per file"
    )
    parser.add_argument("--input_directory", type=str, required=True,
                        help="Input directory path")
    parser.add_argument("--output_directory", type=str, required=True,
                        help="Output directory path")
    parser.add_argument("--objectives", type=str, required=True,
                        help='JSON list of objectives, e.g. \'["sim", "activity"]\'')
    parser.add_argument("--num_workers", type=int, required=True,
                        help="Number of workers")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    objectives = json.loads(args.objectives)

    process_all_files_in_directory(
        args.input_directory,
        args.output_directory,
        objectives,
        args.num_workers
    )


if __name__ == "__main__":
    main()