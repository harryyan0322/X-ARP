# X-ARP: Agentic LLMs for Social Network Generation

This repository implements the **Generate–Detect–Explain–Refine** loop described in  
*Agentic LLMs for Social Network Generation using Explainable Adversarial Re‑prompting*.

**Pipeline (Paper Methodology)**  
1. **Generate**: LLM agents synthesize user profiles, content, and relations.  
2. **Detect**: an explainable bot detector evaluates realism on the mixed network.  
3. **Explain**: extract multi‑level evidence and produce explanations.  
4. **Refine**: convert explanations to re‑prompts to update agent behavior.

The executable pipeline is in `src/main.py`.

## Environment

We use a conda environment named `sc`.  

```bash
conda env create -f sc_env.yml
conda activate sc
```

## Data Preparation

Update `config.yaml` with your local paths. The pipeline expects:

- **User seed data** (for generation):  
  `data.user_path` → e.g. `data/ori_data_complete.json`
- **Detector artifacts** (for explainability):  
  - Model: `src/detector/models_quick/.../best_seed*.pth`  
  - Features: `src/detector/processed_data/merged_cleaned_profiles/`

Optional environment overrides:

```bash
export PIPELINE_USER_PATH=/path/to/ori_data_complete.json
export DETECTOR_MODEL_PATH=/path/to/best_seed.pth
export DETECTOR_DATA_DIR=/path/to/processed_data/merged_cleaned_profiles
```

## Run

```bash
python src/main.py
```

The script is resilient: if data or model artifacts are missing, it skips the corresponding stage rather than failing.

## Outputs

- `output/` — generated network snapshots  
- `explanations/` — explanations and re‑prompt artifacts  
- `logs/` — pipeline logs and summaries

## Notes

- The detector currently assumes a CUDA‑enabled environment; if CUDA is unavailable, the detection stage is skipped.
- API keys and model settings are configured in `config.yaml`.
