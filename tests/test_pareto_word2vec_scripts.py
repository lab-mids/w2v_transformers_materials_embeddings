import tempfile
import unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from tests._helpers import load_module, ensure_dummy_matnexus

ROOT = Path(__file__).resolve().parents[1]


class TestWord2VecParetoScripts(unittest.TestCase):
    def setUp(self):
        ensure_dummy_matnexus()

    def test_word2vec_dataset_preprocess(self):
        mod = load_module(
            "w2v_preprocess_mod",
            ROOT / "02_pareto_prediction" / "Word2Vec" / "script" / "dataset_preprocess.py",
        )

        class DummyPreparer:
            def __init__(self, model_path, property_list):
                self.property_list = property_list

            def add_dataset(self, input_path, output_path, num_workers=1):
                df = pd.read_csv(input_path)
                for prop in self.property_list:
                    df[f"Similarity_to_{prop}"] = 0.5
                df["Material_Vec"] = "[0,0]"
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_path, index=False)

        mod.ProcessPoolExecutor = ThreadPoolExecutor
        mod.DatasetPreparer = DummyPreparer

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "in"
            output_dir = Path(tmpdir) / "out"
            input_dir.mkdir()
            df = pd.DataFrame({"H": [0.5, 0.2], "O": [0.5, 0.8]})
            in_file = input_dir / "Test_material_system.csv"
            df.to_csv(in_file, index=False)

            mod.process_all_files_in_directory(
                str(input_dir),
                str(output_dir),
                model_path="dummy.model",
                property_list=["dielectric", "conductivity"],
                num_workers=1,
            )

            out_file = output_dir / "Test_material_system_with_similarity.csv"
            self.assertTrue(out_file.exists())

    def test_word2vec_dataset_pareto_front(self):
        mod = load_module(
            "w2v_pareto_mod",
            ROOT / "02_pareto_prediction" / "Word2Vec" / "script" / "dataset_pareto_front_cal.py",
        )
        mod.ProcessPoolExecutor = ThreadPoolExecutor

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "in"
            output_dir = Path(tmpdir) / "out"
            input_dir.mkdir()
            df = pd.DataFrame({
                "Similarity_to_a": [0.1, 0.2, 0.3],
                "Similarity_to_b": [0.3, 0.2, 0.1],
            })
            in_file = input_dir / "Test_material_system_with_similarity.csv"
            df.to_csv(in_file, index=False)

            mod.process_all_files_in_directory(
                str(input_dir),
                str(output_dir),
                objectives=["Similarity_to_a", "Similarity_to_b"],
                num_workers=1,
            )

            out_file = output_dir / "Test_material_system_with_similarity_pareto_front.csv"
            self.assertTrue(out_file.exists())


if __name__ == "__main__":
    unittest.main()
