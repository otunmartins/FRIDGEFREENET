# Third-party benchmark systems (non–Bayesian optimization)

These benchmarks are **independent of the MCP server** ([`insulin_ai_mcp_server.py`](../insulin_ai_mcp_server.py)) and of OpenCode tools. They live under [`extern/benchmarks/`](../extern/benchmarks/) (clones are gitignored); thin entry scripts are in [`benchmarks/`](../benchmarks/). Same separation as the in-repo Optuna PSMILES benchmark ([`benchmarks/optuna_psmiles_discovery.py`](../benchmarks/optuna_psmiles_discovery.py)).

## 1. Polymer Generative Models Benchmark (Wisconsin)

| | |
|--|--|
| **Paper** | [Benchmarking study of deep generative models for inverse polymer design](https://doi.org/10.1039/D4DD00395K) — *Digital Discovery* |
| **Code** | [ytl0410/Polymer-Generative-Models-Benchmark](https://github.com/ytl0410/Polymer-Generative-Models-Benchmark) |
| **Method** | Generative models (VAE, AAE, ORGAN, CharRNN, REINVENT, GraphINVENT); RL fine-tuning in later experiments — **not** Bayesian optimization over a surrogate |
| **Weights / data** | Zenodo links in upstream README (MOSES folder, GraphINVENT, RL checkpoints) |

**Clone:** [`extern/benchmarks/polymer-generative-models/README.md`](../extern/benchmarks/polymer-generative-models/README.md) or `bash scripts/clone_external_benchmarks.sh`

**Wrapper:** `python benchmarks/polymer_generative_models_benchmark.py` — verifies clone and MOSES layout.

**Objective:** Metrics are **as defined in the paper**, not insulin-ai’s `discovery_score` (PSMILES + OpenMM) unless you add a custom adapter later.

## 2. IBM logical-agent-driven polymer discovery

| | |
|--|--|
| **Paper** | [Reinforcement Learning with Logical Action-Aware Features for Polymer Discovery](https://research.ibm.com/publications/reinforcement-learning-with-logical-action-aware-features-for-polymer-discovery) — RL4RealLife @ ICML 2021 |
| **Code** | [IBM/logical-agent-driven-polymer-discovery](https://github.com/IBM/logical-agent-driven-polymer-discovery) |
| **Method** | Neuro-symbolic RL (logical action-aware features; DQN-style training in upstream) — **not** Bayesian optimization |

**Clone:** [`extern/benchmarks/ibm-logical-agent-polymer/README.md`](../extern/benchmarks/ibm-logical-agent-polymer/README.md)

**Setup:** `pip install -e md-envs`, unzip `data/polymerDiscovery.zip`, `python scripts/update_pickled_function.py` (see upstream README).

**Wrapper:** `python benchmarks/ibm_polymer_rl_benchmark.py` — runs `python scripts/main.py test -h` as a CLI smoke check when the clone exists.

**License:** Follow upstream `LICENSE` for redistribution.

## Dependencies

Heavy stacks (**PyTorch**, etc.) come from **upstream** `requirements` / conda envs — not pinned in insulin-ai’s base [`pyproject.toml`](../pyproject.toml). Install inside a dedicated venv or conda env if you run full training. See also [`docs/DEPENDENCIES.md`](DEPENDENCIES.md).

## Optional: GLAS (genetic algorithm)

[GLAS](https://github.com/drcassar/glas) ([arXiv:2008.09187](https://arxiv.org/abs/2008.09187)) — GA + ML for optical glasses; same `extern/benchmarks/` pattern if added later.
