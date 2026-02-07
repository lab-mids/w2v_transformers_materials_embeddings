import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from tests._helpers import load_module

ROOT = Path(__file__).resolve().parents[1]


class TestPlotScripts(unittest.TestCase):
    def test_error_fraction_scatter_plot(self):
        mod = load_module(
            "error_fraction_mod",
            ROOT / "04_plots" / "scripts" / "error_fraction_scatter.py",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            out_fig = Path(tmpdir) / "fig.pdf"
            df = pd.DataFrame({
                "method": ["W2V", "Qwen"],
                "error (%)": [1.2, -0.4],
                "fraction (%)": [10, 20],
            })
            df.to_csv(csv_path, index=False)

            plotter = mod.ErrorFractionScatterPlotter(csv_path)
            plotter.plot_error_vs_fraction(outfile=out_fig)
            self.assertTrue(out_fig.exists())

    def test_fraction_retained_plot(self):
        mod = load_module(
            "fraction_retained_mod",
            ROOT / "04_plots" / "scripts" / "fraction_retained_plot.py",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            out_fig = Path(tmpdir) / "fig.pdf"
            df = pd.DataFrame({
                "method": ["W2V", "Qwen"],
                "fraction (%)": [10, 20],
            })
            df.to_csv(csv_path, index=False)

            plotter = mod.FractionRetainedPlotter(csv_path)
            plotter.plot_fraction_bar_points(outfile=out_fig)
            self.assertTrue(out_fig.exists())

    def test_error_heatmap_plot(self):
        mod = load_module(
            "error_heatmap_mod",
            ROOT / "04_plots" / "scripts" / "error_heatmap_plot.py",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            out_fig = Path(tmpdir) / "fig.pdf"
            df = pd.DataFrame({
                "material_system": ["Ag-Pd"],
                "overpotential (mV)": [100],
                "method": ["W2V"],
                "error (%)": [2.5],
            })
            df.to_csv(csv_path, index=False)

            plotter = mod.ErrorHeatmapPlotter(csv_path)
            plotter.plot_error_heatmap(outfile=out_fig)
            self.assertTrue(out_fig.exists())

    def test_material_system_panels(self):
        mod = load_module(
            "material_system_mod",
            ROOT / "04_plots" / "scripts" / "material_system_panels.py",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            method_dir = base / "Word2Vec"
            sim_dir = method_dir / "material_systems_with_similarities"
            pareto_dir = method_dir / "material_systems_pareto_front"
            sim_dir.mkdir(parents=True)
            pareto_dir.mkdir(parents=True)
            out_dir = base / "out"

            df = pd.DataFrame({
                "x": [0.1, 0.2],
                "y": [0.3, 0.4],
                "Current_at_100mV": [1.0, 2.0],
            })
            sim_file = sim_dir / "Ag_Pd_material_system_with_similarity.csv"
            pareto_file = pareto_dir / "Ag_Pd_material_system_with_similarity_pareto_front.csv"
            df.to_csv(sim_file, index=False)
            df.head(1).to_csv(pareto_file, index=False)

            plotter = mod.MaterialSystemPlotter(
                method_dirs=[str(method_dir)],
                output_dir=str(out_dir),
                grid_shape=(1, 2),
                dpi=50,
            )
            plotter.run()

            outputs = list(out_dir.glob("*.pdf"))
            self.assertTrue(outputs)

    def test_word_embedding_distribution_plot(self):
        mod = load_module(
            "word_embedding_mod",
            ROOT / "04_plots" / "scripts" / "word_embedding_distribution_plot.py",
        )

        class DummyTSNE:
            def __init__(self, *args, **kwargs):
                pass

            def fit_transform(self, X):
                X = np.asarray(X)
                if X.shape[1] >= 2:
                    return X[:, :2]
                return np.zeros((X.shape[0], 2))

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            model_path = base / "model.w2v"
            data_path = base / "Ag_Pd_material_system.csv"
            out_dir = base / "out"
            done_flag = base / "done.txt"

            sentences = [["H", "O", "dielectric", "conductivity"], ["H", "O"], ["dielectric", "conductivity"]]
            model = Word2Vec(sentences=sentences, vector_size=10, min_count=1, epochs=5)
            model.save(str(model_path))

            df = pd.DataFrame({"H": [0.5, 0.2], "O": [0.5, 0.8]})
            df.to_csv(data_path, index=False)

            with patch.object(mod, "TSNE", DummyTSNE):
                mod.main(
                    model_path=str(model_path),
                    input_files=[str(data_path)],
                    output_dir=str(out_dir),
                    output_basename="materials_embedding",
                    done_flag=str(done_flag),
                    tsne_perplexity=2.0,
                    tsne_n_iter=250,
                )

            self.assertTrue(done_flag.exists())
            self.assertTrue((out_dir / "materials_embedding.pdf").exists())


if __name__ == "__main__":
    unittest.main()
