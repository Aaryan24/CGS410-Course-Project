# CGS410 Course Project

This repository contains the final implementation code for the CGS410 course project on syntactic attention heads, dependency recovery, and prediction efficiency in mBERT and GPT-2 medium. Report-generation scripts and draft-building utilities are intentionally excluded.

The code is organized so that the study can be reproduced from the implementation side: multilingual mBERT dependency-head scoring, English mBERT retained-head masking/ablation, GPT-2 dependency scoring, GPT-2 retained-head sufficiency, GPT-2 necessity tests, and the GPT-2 causal-ceiling diagnostic.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Main entry points:

- `mbert_dependency_head_analysis.py`: multilingual mBERT dependency-head scoring over UD treebanks.
- `run_mbert_multilingual_dependency_batch.py`: batch runner for multilingual mBERT dependency-head experiments.
- `mbert_prediction_efficiency.py`: mBERT head-ranking, masking, sufficiency, and ablation experiments.
- `gpt2_prediction_efficiency.py`: GPT-2 medium retained-head prediction-efficiency experiments.
- `gpt2_dependency_head_analysis.py`, `gpt2_phase2_sufficiency.py`, `gpt2_necessity_test.py`, and `gpt2_syntax_necessity.py`: GPT-2 dependency and prediction tests.
- `compute_gpt2_causal_ceiling.py`: causal recoverability diagnostic for GPT-2.
- `multilingual_language_config.json`: UD treebank URLs and run names for the multilingual mBERT dependency-head runs.
- `study_graphs/`: graph assets from the final study, included so results and report figures can be reviewed directly from the repository.

## Reproduction Notes

The scripts download Universal Dependencies files from the public UD GitHub mirrors when needed. Transformer models are loaded through Hugging Face `transformers`.

Example multilingual mBERT dependency-head run:

```bash
python run_mbert_multilingual_dependency_batch.py --languages en_ewt,hi_hdtb,ur_udtb,fi_tdt,id_gsd --device auto
```

Example GPT-2 phase-1 dependency-head run:

```bash
python gpt2_dependency_head_analysis.py --model-name gpt2-medium --split dev --limit 0
```

The prediction-efficiency scripts expect the relevant phase-1 ranking outputs to exist in `results/`. Large model runs can take a long time and may need a GPU or Apple Silicon MPS device.
