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

**Directory Map (Where To Work)**
- `01_word2vec_model`: Collect papers, process text, train word2vec.
- `02_pareto_prediction`: Compute similarities + Pareto fronts for multiple embedding models.
- `03_analysis`: Aggregate similarity and Pareto metrics into a master CSV.
- `04_plots`: Plot figures from the analysis CSV.

**01_word2vec_model**
- Config: `01_word2vec_model/config.yaml`.
- Snakemake: `01_word2vec_model/Snakefile`.
- Notes:
  - You must create `01_word2vec_model/pybliometrics/` and provide API keys in `01_word2vec_model/pybliometrics/pybliometrics.cfg`.
  - The configuration file for the workflow in the root folder of the 01_* step`config.yaml` has a placeholder `your_scopy_api_key` that must be replaced with your own key.
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
  - Each config expects an `input_directory` (default `material_systems`).
  - You need to create and populate that folder with material-system CSVs before running.
- API key requirement (Qwen/Qwen_Full):
  - The Qwen workflows call a Blablador embeddings endpoint and require `BLABLADOR_API_KEY` in the environment.
  - Optional overrides: `BLABLADOR_BASE_URL` and `BLABLADOR_MODEL`.
  - `Qwen_Full/config.yaml` also contains an `api_key` field used by its Snakefile; prefer exporting the env var to avoid committing secrets.
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
