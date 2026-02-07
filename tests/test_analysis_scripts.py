import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from tests._helpers import load_module

ROOT = Path(__file__).resolve().parents[1]


class TestAnalysisScripts(unittest.TestCase):
    def test_analysis_main_creates_output(self):
        analysis_mod = load_module(
            "analysis_mod", ROOT / "03_analysis" / "script" / "analysis.py"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            root_dir = base / "root"
            pareto_dir = base / "pareto"
            out_dir = base / "analysis_results"
            root_dir.mkdir()
            pareto_dir.mkdir()

            df = pd.DataFrame({
                "H": [0.5, 0.2, 0.1],
                "O": [0.5, 0.8, 0.9],
                "Current_at_100mV": [1.0, -2.0, 0.5],
            })
            sim_file = root_dir / "Ag_Pd_material_system.csv"
            pareto_file = pareto_dir / "Ag_Pd_material_system.csv"
            df.to_csv(sim_file, index=False)
            df.head(2).to_csv(pareto_file, index=False)

            cfg = {
                "output_dir": str(out_dir),
                "methods": [
                    {
                        "name": "W2V",
                        "root": str(root_dir),
                        "pareto_root": str(pareto_dir),
                    }
                ],
            }
            cfg_path = base / "config.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg))

            argv = ["analysis.py", "--config", str(cfg_path)]
            with patch.object(sys, "argv", argv):
                analysis_mod.main()

            out_file = out_dir / "Ag_Pd_material_system_analysis.csv"
            self.assertTrue(out_file.exists())

    def test_aggregate_similarity_analysis(self):
        agg_mod = load_module(
            "agg_mod", ROOT / "03_analysis" / "script" / "aggregate_similarity_analysis.py"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            analysis_dir = Path(tmpdir) / "analysis_results"
            analysis_dir.mkdir()
            sample = pd.DataFrame({"method": ["W2V"], "error (%)": [1.0]})
            sample.to_csv(analysis_dir / "Ag_Pd_material_system_analysis.csv", index=False)

            out_path = analysis_dir / "master_analysis.csv"
            argv = [
                "aggregate_similarity_analysis.py",
                "--analysis-dir",
                str(analysis_dir),
                "--output",
                str(out_path),
            ]
            with patch.object(sys, "argv", argv):
                agg_mod.main()

            self.assertTrue(out_path.exists())


if __name__ == "__main__":
    unittest.main()
