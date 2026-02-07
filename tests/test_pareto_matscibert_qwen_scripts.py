import os
import tempfile
import unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from tests._helpers import load_module, ensure_dummy_torch_and_transformers

ROOT = Path(__file__).resolve().parents[1]


def _make_stub_embed(df, concepts=None, **kwargs):
    out = df.copy()
    out["Material_Vec"] = [[0.0, 0.0] for _ in range(len(df))]
    if concepts:
        for label in concepts:
            out[f"Similarity_to_{label}"] = 0.1
    return out


def _run_preprocess(module_path: Path, module_name: str, output_suffix: str):
    mod = load_module(module_name, module_path)
    mod.ProcessPoolExecutor = ThreadPoolExecutor
    mod.matscibert_embed_and_similarity = _make_stub_embed

    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "in"
        output_dir = Path(tmpdir) / "out"
        input_dir.mkdir()
        df = pd.DataFrame({"H": [0.5, 0.2], "O": [0.5, 0.8]})
        in_file = input_dir / "Test_material_system.csv"
        df.to_csv(in_file, index=False)

        mod.process_all_files_in_directory(
            input_directory=str(input_dir),
            output_directory=str(output_dir),
            filename_suffix="_material_system.csv",
            num_workers=1,
            prompt_style="composition",
            extra_tags=None,
            concepts={"dielectric": "x"},
            batch_size=2,
            device="cpu",
            max_length=16,
            output_suffix=output_suffix,
        )

        out_file = output_dir / f"Test_material_system{output_suffix}.pkl"
        assert out_file.exists(), f"Expected output not found: {out_file}"
        df_out = pd.read_pickle(out_file)
        assert "Material_Vec" in df_out.columns


def _run_pareto(module_path: Path, module_name: str, filename_suffix: str):
    mod = load_module(module_name, module_path)
    mod.ProcessPoolExecutor = ThreadPoolExecutor

    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "in"
        output_dir = Path(tmpdir) / "out"
        input_dir.mkdir()
        df = pd.DataFrame({
            "Similarity_to_a": [0.1, 0.2, 0.3],
            "Similarity_to_b": [0.3, 0.2, 0.1],
        })
        in_file = input_dir / f"Test_material_system{filename_suffix}"
        df.to_pickle(in_file)

        mod.process_all_files_in_directory(
            str(input_dir),
            str(output_dir),
            objectives=["Similarity_to_a", "Similarity_to_b"],
            num_workers=1,
            filename_suffix=filename_suffix,
            output_format="csv",
        )

        df_key = Path(in_file).stem
        out_file = output_dir / f"{df_key}_pareto_front.csv"
        assert out_file.exists(), f"Expected output not found: {out_file}"


class TestMatSciBERTAndQwenScripts(unittest.TestCase):
    def setUp(self):
        ensure_dummy_torch_and_transformers()

    def test_matscibert_preprocess(self):
        _run_preprocess(
            ROOT / "02_pareto_prediction" / "MatSciBERT" / "script" / "dataset_preprocess.py",
            "matscibert_preprocess_mod",
            "_with_matscibert",
        )

    def test_matscibert_full_preprocess(self):
        _run_preprocess(
            ROOT / "02_pareto_prediction" / "MatSciBERT_Full" / "script" / "dataset_preprocess.py",
            "matscibert_full_preprocess_mod",
            "_with_matscibert",
        )

    def test_qwen_preprocess(self):
        os.environ.setdefault("BLABLADOR_API_KEY", "test-key")
        _run_preprocess(
            ROOT / "02_pareto_prediction" / "Qwen" / "script" / "dataset_preprocess.py",
            "qwen_preprocess_mod",
            "_with_qwen",
        )

    def test_qwen_full_preprocess(self):
        os.environ.setdefault("BLABLADOR_API_KEY", "test-key")
        _run_preprocess(
            ROOT / "02_pareto_prediction" / "Qwen_Full" / "script" / "dataset_preprocess.py",
            "qwen_full_preprocess_mod",
            "_with_qwen",
        )

    def test_matscibert_pareto(self):
        _run_pareto(
            ROOT / "02_pareto_prediction" / "MatSciBERT" / "script" / "dataset_pareto_front_cal.py",
            "matscibert_pareto_mod",
            "_with_matscibert.pkl",
        )

    def test_matscibert_full_pareto(self):
        _run_pareto(
            ROOT / "02_pareto_prediction" / "MatSciBERT_Full" / "script" / "dataset_pareto_front_cal.py",
            "matscibert_full_pareto_mod",
            "_with_matscibert.pkl",
        )

    def test_qwen_pareto(self):
        _run_pareto(
            ROOT / "02_pareto_prediction" / "Qwen" / "script" / "dataset_pareto_front_cal.py",
            "qwen_pareto_mod",
            "_with_qwen.pkl",
        )

    def test_qwen_full_pareto(self):
        _run_pareto(
            ROOT / "02_pareto_prediction" / "Qwen_Full" / "script" / "dataset_pareto_front_cal.py",
            "qwen_full_pareto_mod",
            "_with_qwen.pkl",
        )


if __name__ == "__main__":
    unittest.main()
