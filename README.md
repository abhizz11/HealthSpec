# HealthSpec

Adaptive branching-tree speculative decoding for biomedical prompts. A tabular
contextual selector chooses among 24 tree shapes using draft entropy and previous
accepted draft length. Language-model weights are unchanged.

## Source files

| File | Purpose |
|---|---|
| `adaptive_tree_sd.py` | Adaptive selection policy and generation |
| `fixed_tree_sd.py` | Tree construction, verification and cache handling |
| `tree_configs.py` | Tree configurations and fixed baseline |
| `benchmark_final_pubmedqa.py` | Final benchmark runner and checkpoint protocol |
| `benchmark_pubmedqa.py` | Prompt formatting and benchmark statistics |
| `benchmark_branching_pubmedqa.py` | Benchmark helpers and prior-prompt audit |
| `plot_final_pubmedqa.py` | Validate local experiment records and export figures |

## Setup

Use a separate Python environment and install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Torch and Transformers versions reflect the recorded environment. Other
requirements are unpinned. Install a CUDA-enabled PyTorch build appropriate for
benchmarking on your GPU.

## Local experiments

Experiment records, trained policies, results, paper artifacts, model weights and
caches are excluded from this repository. Keep reproduction results separately.

The benchmark scripts currently depend on local model snapshots, dataset files,
metadata and saved protocol paths. Configure those inputs before running inference;
a fresh clone is not a complete standalone reproduction package.

With a completed local experiment directory and its required checkpoint blocks,
reference tokens and protocol, regenerate reports using:

```powershell
python plot_final_pubmedqa.py PATH_TO_LOCAL_EXPERIMENT
```

This command validates saved measurements and writes reports and figures into
that directory. It does not run inference.

See [REPOSITORY_GUIDE.md](REPOSITORY_GUIDE.md) for publication commands.
