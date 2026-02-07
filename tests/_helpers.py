import importlib.util
import sys
import types
from pathlib import Path


def load_module(module_name: str, path: str):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def ensure_dummy_module(name: str, attrs: dict | None = None):
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    if attrs:
        for key, value in attrs.items():
            setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


def ensure_dummy_torch_and_transformers():
    # torch
    if "torch" not in sys.modules:
        try:
            import importlib.util
            if importlib.util.find_spec("torch") is not None:
                import torch  # noqa: F401
        except Exception:
            # Fall back to dummy torch if real import fails
            pass
    if "torch" not in sys.modules:
        torch_mod = ensure_dummy_module("torch")
        nn_mod = ensure_dummy_module("torch.nn")
        functional_mod = ensure_dummy_module("torch.nn.functional")
        torch_mod.nn = nn_mod
        nn_mod.functional = functional_mod
        # minimal attributes used in code paths
        torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
        torch_mod.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
        torch_mod.Tensor = object
        torch_mod.no_grad = types.SimpleNamespace(__enter__=lambda *a, **k: None, __exit__=lambda *a, **k: None)

    # transformers
    if "transformers" not in sys.modules:
        try:
            import importlib.util
            if importlib.util.find_spec("transformers") is not None:
                import transformers  # noqa: F401
        except Exception:
            # Fall back to dummy transformers if real import fails
            pass
    if "transformers" not in sys.modules:
        class _DummyTokenizer:
            def __call__(self, *args, **kwargs):
                return {"input_ids": [], "attention_mask": []}

            @staticmethod
            def from_pretrained(*args, **kwargs):
                return _DummyTokenizer()

        class _DummyModel:
            def __call__(self, *args, **kwargs):
                class _Out:
                    last_hidden_state = []
                return _Out()

            def to(self, *args, **kwargs):
                return self

            def eval(self):
                return self

            @staticmethod
            def from_pretrained(*args, **kwargs):
                return _DummyModel()

        ensure_dummy_module(
            "transformers",
            attrs={"AutoTokenizer": _DummyTokenizer, "AutoModel": _DummyModel},
        )


def ensure_dummy_matnexus():
    if "matnexus" in sys.modules:
        return sys.modules["matnexus"]

    class _DummyPaperCollector:
        class ScopusDataSource:
            def __init__(self, *args, **kwargs):
                pass

        class ArxivDataSource:
            def __init__(self, *args, **kwargs):
                pass

        class MultiSourcePaperCollector:
            def __init__(self, *args, **kwargs):
                self.results = None

            @staticmethod
            def build_query(**kwargs):
                return kwargs

            def collect_papers(self):
                return None

    class _DummyTextProcessor:
        class TextProcessor:
            def __init__(self, *args, **kwargs):
                self.processed_df = None

    class _DummyVecGenerator:
        class Corpus:
            def __init__(self, *args, **kwargs):
                self.sentences = []

        class Word2VecModel:
            def __init__(self, *args, **kwargs):
                pass

            @staticmethod
            def load(*args, **kwargs):
                return _DummyVecGenerator.Word2VecModel()

        class MaterialSimilarityCalculator:
            def __init__(self, *args, **kwargs):
                pass

            def calculate_similarity_from_dataframe(self, df, elements, target_property, add_experimental_indicator=False):
                out = df.copy()
                out["Similarity"] = 0.0
                out["Material_Vec"] = "[0,0]"
                return out

    return ensure_dummy_module(
        "matnexus",
        attrs={
            "PaperCollector": _DummyPaperCollector,
            "TextProcessor": _DummyTextProcessor,
            "VecGenerator": _DummyVecGenerator,
        },
    )
