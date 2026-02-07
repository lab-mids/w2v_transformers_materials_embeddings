import argparse
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from matnexus import VecGenerator
import json

# Predefined list of periodic table elements
PERIODIC_TABLE_ELEMENTS = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P',
    'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se',
    'Br', 'Kr', 'Rb', 'Sr', 'Y',
    'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',
    'Xe', 'Cs', 'Ba', 'La', 'Ce',
    'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf',
    'Ta', 'W', 'Re', 'Os', 'Ir',
    'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
    'U', 'Np', 'Pu', 'Am', 'Cm',
    'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og'
]


class DatasetPreparer:
    def __init__(self, model_path, property_list):
        self.model = VecGenerator.Word2VecModel.load(model_path)
        self.property_list = property_list
        self.calculator = VecGenerator.MaterialSimilarityCalculator(self.model)

    def extract_elements_from_columns(self, columns):
        """Extract element symbols from the column names by matching with the periodic table."""
        elements = [col for col in columns if col in PERIODIC_TABLE_ELEMENTS]
        return elements

    def process_chunk(self, chunk, elements):
        result_df = chunk.copy()
        for prop in self.property_list:
            temp_df = self.calculator.calculate_similarity_from_dataframe(chunk,
                                                                          elements,
                                                                          target_property=[
                                                                              prop],
                                                                          add_experimental_indicator=False)
            result_df[f'Similarity_to_{prop}'] = temp_df['Similarity']
        result_df['Material_Vec'] = temp_df['Material_Vec']
        return result_df

    def add_dataset(self, data_path, output_path, num_workers=4):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        data = pd.read_csv(data_path)
        # Extract elements from column names
        elements = self.extract_elements_from_columns(data.columns)
        if not elements:
            raise ValueError(
                f"No valid element symbols found in the columns of {data_path}")

        chunk_size = len(data) // num_workers
        chunks = [data.iloc[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.process_chunk, chunk, elements) for chunk in
                       chunks]
            results = [future.result() for future in futures]

        df = pd.concat(results)
        df.to_csv(output_path, index=False)


def load_processed_files(log_path):
    if os.path.exists(log_path):
        with open(log_path, 'r') as file:
            processed_files = set(line.strip() for line in file)
    else:
        processed_files = set()
    return processed_files


def log_processed_file(log_path, filename):
    with open(log_path, 'a') as file:
        file.write(filename + '\n')


def process_file(filename, input_directory, output_directory, model_path, property_list,
                 log_path, num_workers=4):
    preparer = DatasetPreparer(model_path, property_list)
    input_path = os.path.join(input_directory, filename)
    base_filename = filename.rsplit('.', 1)[0]
    output_filename = f'{base_filename}_with_similarity.csv'
    output_path = os.path.join(output_directory, output_filename)

    preparer.add_dataset(input_path, output_path, num_workers)
    log_processed_file(log_path, filename)


def process_all_files_in_directory(input_directory, output_directory, model_path,
                                   property_list, num_workers=4):
    log_path = os.path.join(output_directory, 'processed_files.log')
    processed_files = load_processed_files(log_path)

    filenames = [f for f in os.listdir(input_directory) if
                 f.endswith('_material_system.csv')]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_file, filename, input_directory, output_directory,
                            model_path, property_list, log_path, num_workers)
            for filename in filenames if filename not in processed_files
        ]
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f'Error processing file: {e}')


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess dataset files")
    parser.add_argument("--input_directory", type=str, required=True,
                        help="Input directory path")
    parser.add_argument("--output_directory", type=str, required=True,
                        help="Output directory path")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model file")
    parser.add_argument("--property_list", type=str, required=True,
                        help="List of properties")
    parser.add_argument("--num_workers", type=int, required=True,
                        help="Number of workers")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    property_list = json.loads(args.property_list)

    process_all_files_in_directory(args.input_directory, args.output_directory,
                                   args.model_path, property_list, args.num_workers)


if __name__ == "__main__":
    main()