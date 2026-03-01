# From Word2Vec to Transformers: Text-Derived Composition Embeddings for Filtering Combinatorial Electrocatalysts

Code repository for the paper “From Word2Vec to Transformers: Text-Derived Composition Embeddings for Filtering Combinatorial Electrocatalysts”.

This repo is organized as a sequence of Snakemake workflows that build a word2vec model, compute material-system similarities/Pareto fronts for different embedding models, aggregate analyses, and generate plots.

**Quick Start**
1. Create the environment.
2. Run the workflows in order: `01_word2vec_model` → `02_pareto_prediction` → `03_analysis` → `04_plots`.

Quick tests: `./tests/run_tests.sh`.

**Environment**
- Conda env file: `word_embedding_transformer.yml` (exported and normalized for cross-platform use).
- Create and activate:

```bash
conda env create -f word_embedding_transformer.yml
conda activate word_embedding_transformer
```

- Install the spaCy English model required by `01_word2vec_model/script/process_papers.py`:

```bash
python -m spacy download en_core_web_sm
```

- MatSciBERT note:
  - No separate MatSciBERT installation is required; `torch` and `transformers` in `word_embedding_transformer.yml` are sufficient.
  - `MatSciBERT` and `MatSciBERT_Full` workflows download `m3rg-iitd/matscibert` automatically on first run via Hugging Face, so internet access is required at least once.

**Directory Map (Where To Work)**
- `01_word2vec_model`: Collect papers, process text, train word2vec.
- `02_pareto_prediction`: Compute similarities + Pareto fronts for multiple embedding models.
- `03_analysis`: Aggregate similarity and Pareto metrics into a master CSV.
- `04_plots`: Plot figures from the analysis CSV.

**01_word2vec_model**
- Config: `01_word2vec_model/config.yaml`.
- Snakemake: `01_word2vec_model/Snakefile`.
- Notes:
  - Set Scopus API key(s) in `01_word2vec_model/config.yaml` under `pybliometrics_config -> Authentication -> APIKey`.
  - Create `01_word2vec_model/pybliometrics/` before the first run; the workflow will create/update `01_word2vec_model/pybliometrics/pybliometrics.cfg` from `config.yaml`.
  - Ensure `en_core_web_sm` is installed in the active environment before running the workflow.
- Run:

```bash
cd 01_word2vec_model
snakemake -s Snakefile -j 1
```

**02_pareto_prediction**
- Subdirectories correspond to different embedding models:
  - `Word2Vec`, `MatSciBERT`, `MatSciBERT_Full`, `Qwen`, `Qwen_Full`.
- Each subdir has its own `Snakefile` and `config.yaml`.
- Required inputs:
  - Each config expects an `input_directory` (default `../material_systems`), so all methods share `02_pareto_prediction/material_systems`.
  - Create and populate `02_pareto_prediction/material_systems` with material-system CSVs before running.
  - Input filenames should end with `_material_system.csv` (for example `Ni_Pd_Pt_Ru_material_system.csv`).
  - Each CSV should contain composition columns named by element symbols (for example `Ni`, `Pd`, `Pt`, `Ru`) and at least one current column such as `Current_at_100mV` for downstream analysis.
- Reference datasets (recommended format examples):
  - ORR dataset: [https://doi.org/10.5281/zenodo.13992986](https://doi.org/10.5281/zenodo.13992986)
  - HER dataset: [https://doi.org/10.5281/zenodo.14959252](https://doi.org/10.5281/zenodo.14959252)
  - OER dataset (Ni-Pd-Pt-Ru): [https://doi.org/10.5281/zenodo.14891704](https://doi.org/10.5281/zenodo.14891704)
- API key requirement (Qwen/Qwen_Full):
  - Both workflows require `BLABLADOR_API_KEY` at runtime.
  - `Qwen` and `Qwen_Full` both read `blablador` settings from their `config.yaml` and export them in the workflow.
  - Filling `blablador.api_key` in either `Qwen/config.yaml` or `Qwen_Full/config.yaml` works.
  - Optional overrides: `BLABLADOR_BASE_URL` and `BLABLADOR_MODEL`.
  - Recommended usage for both workflows is still shell env vars (to avoid storing secrets in repo files), for example:

```bash
export BLABLADOR_API_KEY="your_api_key"
export BLABLADOR_BASE_URL="https://api.helmholtz-blablador.fz-juelich.de/v1"
export BLABLADOR_MODEL="alias-embeddings"
```
- Run one model:

```bash
cd 02_pareto_prediction/Word2Vec
snakemake -s Snakefile -j 1
```

- Run all models:

```bash
cd 02_pareto_prediction
./run_all.sh
```

**03_analysis**
- Config: `03_analysis/config.yaml`.
- Inputs: expects the `material_systems_with_similarities` and `material_systems_pareto_front` outputs from `02_pareto_prediction`.
- Run:

```bash
cd 03_analysis
snakemake -s Snakefile -j 1
```

- Output:
  - `03_analysis/analysis_results/master_analysis.csv`.

**04_plots**
- Configs in `04_plots/configs/` point to `03_analysis/analysis_results/master_analysis.csv` by default.
- Run all plots:

```bash
cd 04_plots
./run_all.sh
```

- Or run a single plot workflow:

```bash
cd 04_plots
snakemake -s workflows/error_plot.smk -j 1
```

**Tests**
- Run the full test suite:

```bash
./tests/run_tests.sh
```

**License**

GNU GPL v3.0 (see `LICENSE`).
