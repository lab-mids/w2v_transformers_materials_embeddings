import os
import unittest
from pathlib import Path

from tests._helpers import load_module, ensure_dummy_matnexus, ensure_dummy_torch_and_transformers

ROOT = Path(__file__).resolve().parents[1]

SCRIPT_PATHS = [
    ROOT / "01_word2vec_model" / "script" / "collect_papers.py",
    ROOT / "01_word2vec_model" / "script" / "process_papers.py",
    ROOT / "01_word2vec_model" / "script" / "generate_word2vec.py",
    ROOT / "02_pareto_prediction" / "Word2Vec" / "script" / "dataset_preprocess.py",
    ROOT / "02_pareto_prediction" / "Word2Vec" / "script" / "dataset_pareto_front_cal.py",
    ROOT / "02_pareto_prediction" / "MatSciBERT" / "script" / "dataset_preprocess.py",
    ROOT / "02_pareto_prediction" / "MatSciBERT" / "script" / "dataset_pareto_front_cal.py",
    ROOT / "02_pareto_prediction" / "MatSciBERT_Full" / "script" / "dataset_preprocess.py",
    ROOT / "02_pareto_prediction" / "MatSciBERT_Full" / "script" / "dataset_pareto_front_cal.py",
    ROOT / "02_pareto_prediction" / "Qwen" / "script" / "dataset_preprocess.py",
    ROOT / "02_pareto_prediction" / "Qwen" / "script" / "dataset_pareto_front_cal.py",
    ROOT / "02_pareto_prediction" / "Qwen_Full" / "script" / "dataset_preprocess.py",
    ROOT / "02_pareto_prediction" / "Qwen_Full" / "script" / "dataset_pareto_front_cal.py",
    ROOT / "03_analysis" / "script" / "analysis.py",
    ROOT / "03_analysis" / "script" / "aggregate_similarity_analysis.py",
    ROOT / "04_plots" / "scripts" / "error_fraction_scatter.py",
    ROOT / "04_plots" / "scripts" / "fraction_retained_plot.py",
    ROOT / "04_plots" / "scripts" / "error_heatmap_plot.py",
    ROOT / "04_plots" / "scripts" / "material_system_panels.py",
    ROOT / "04_plots" / "scripts" / "word_embedding_distribution_plot.py",
]


class TestScriptImports(unittest.TestCase):
    def test_all_script_files_importable(self):
        ensure_dummy_matnexus()
        ensure_dummy_torch_and_transformers()
        os.environ.setdefault("BLABLADOR_API_KEY", "test-key")

        for path in SCRIPT_PATHS:
            with self.subTest(script=str(path)):
                self.assertTrue(path.exists(), f"Missing script: {path}")
                mod_name = "script_import_" + "_".join(path.parts[-4:]).replace(".", "_").replace("-", "_")
                load_module(mod_name, path)

    def test_run_all_scripts_reference_valid_paths(self):
        run_all = ROOT / "02_pareto_prediction" / "run_all.sh"
        self.assertTrue(run_all.exists())
        base = run_all.parent
        content = run_all.read_text()
        targets = []
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("snakemake -s "):
                parts = line.split()
                if "-s" in parts:
                    idx = parts.index("-s")
                    targets.append(parts[idx + 1])
        for target in targets:
            self.assertTrue((base / target).exists(), f"Missing target: {target}")

        plot_runner = ROOT / "04_plots" / "run_all.sh"
        self.assertTrue(plot_runner.exists())
        plot_base = plot_runner.parent
        content = plot_runner.read_text()
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("\"workflows/"):
                wf = line.strip('"')
                self.assertTrue((plot_base / wf).exists(), f"Missing workflow: {wf}")


if __name__ == "__main__":
    unittest.main()
