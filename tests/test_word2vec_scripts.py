import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from tests._helpers import load_module, ensure_dummy_matnexus

ROOT = Path(__file__).resolve().parents[1]


class TestWord2VecScripts(unittest.TestCase):
    def setUp(self):
        ensure_dummy_matnexus()

    def test_update_pybliometrics_config_creates_file(self):
        utils_mod = load_module(
            "utils_mod", ROOT / "01_word2vec_model" / "utils.py"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "pybliometrics.cfg"
            new_cfg = {
                "Authentication": {"APIKey": ["abc", "def"]},
                "Requests": {"Timeout": 10, "Retries": 1},
            }
            utils_mod.update_pybliometrics_config(str(cfg_path), new_cfg)
            self.assertTrue(cfg_path.exists())
            content = cfg_path.read_text()
            self.assertIn("APIKey = abc,def", content)

    def test_collect_papers_main_writes_csv(self):
        collect_mod = load_module(
            "collect_papers_mod", ROOT / "01_word2vec_model" / "script" / "collect_papers.py"
        )
        dummy_df = pd.DataFrame({"title": ["paper"], "year": [2024]})

        class DummyCollector:
            def __init__(self, sources, query):
                self.sources = sources
                self.query = query
                self.results = dummy_df

            @staticmethod
            def build_query(**kwargs):
                return kwargs

            def collect_papers(self):
                return None

        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = Path(tmpdir) / "out.csv"
            argv = [
                "collect_papers.py",
                "--config_path",
                str(Path(tmpdir) / "cfg.ini"),
                "--keywords",
                "electrocatalyst",
                "--endyear",
                "2024",
                "--openaccess",
                "True",
                "--output_path",
                str(out_csv),
            ]
            with patch.object(sys, "argv", argv):
                with patch.object(collect_mod.spc, "ScopusDataSource", lambda config_path: ("scopus", config_path)):
                    with patch.object(collect_mod.spc, "ArxivDataSource", lambda: "arxiv"):
                        with patch.object(collect_mod.spc, "MultiSourcePaperCollector", DummyCollector):
                            collect_mod.main()
            self.assertTrue(out_csv.exists())
            df = pd.read_csv(out_csv)
            self.assertEqual(df.shape[0], 1)

    def test_process_papers_main_writes_csv(self):
        proc_mod = load_module(
            "process_papers_mod", ROOT / "01_word2vec_model" / "script" / "process_papers.py"
        )

        class DummyTextProcessor:
            def __init__(self, data):
                self.processed_df = data.assign(processed=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            in_csv = Path(tmpdir) / "in.csv"
            out_csv = Path(tmpdir) / "out.csv"
            pd.DataFrame({"text": ["a", "b"]}).to_csv(in_csv, index=False)
            argv = [
                "process_papers.py",
                "--input_path",
                str(in_csv),
                "--output_path",
                str(out_csv),
            ]
            with patch.object(sys, "argv", argv):
                with patch.object(proc_mod.nltk, "download", lambda *args, **kwargs: True):
                    with patch.object(proc_mod.TextProcessor, "TextProcessor", DummyTextProcessor):
                        proc_mod.main()
            self.assertTrue(out_csv.exists())
            df = pd.read_csv(out_csv)
            self.assertIn("processed", df.columns)

    def test_generate_word2vec_main_writes_model(self):
        gen_mod = load_module(
            "generate_word2vec_mod", ROOT / "01_word2vec_model" / "script" / "generate_word2vec.py"
        )

        class DummyCorpus:
            def __init__(self, df):
                self.sentences = [["a", "b"], ["b", "c"]]

        class DummyWord2Vec:
            def __init__(self, sentences):
                self.sentences = sentences
                self.fitted = False

            def fit(self, **kwargs):
                self.fitted = True

            def save(self, path):
                Path(path).write_text("ok")

        with tempfile.TemporaryDirectory() as tmpdir:
            in_csv = Path(tmpdir) / "in.csv"
            model_path = Path(tmpdir) / "model.bin"
            pd.DataFrame({"text": ["a b", "b c"]}).to_csv(in_csv, index=False)
            argv = [
                "generate_word2vec.py",
                "--input_path",
                str(in_csv),
                "--model_path",
                str(model_path),
            ]
            with patch.object(sys, "argv", argv):
                with patch.object(gen_mod.VecGenerator, "Corpus", DummyCorpus):
                    with patch.object(gen_mod.VecGenerator, "Word2VecModel", DummyWord2Vec):
                        gen_mod.main()
            self.assertTrue(model_path.exists())


if __name__ == "__main__":
    unittest.main()
