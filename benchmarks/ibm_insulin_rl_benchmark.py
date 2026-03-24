#!/usr/bin/env python3
"""
IBM Logical-Agent RL benchmark adapted for the insulin-ai evaluation pipeline.

Trains and tests DQN or PPO policies from ``stable-baselines3`` on the
``LogicalInsulinPSMILESEnv`` Gym environment, which uses the **same**
evaluation pipeline as the agentic MCP ``evaluate_psmiles`` tool:

    MDSimulator.evaluate_candidates
    → PropertyExtractor.extract_feedback
    → scoring.composite_screening_score / discovery_score

IBM's optimization loop (neuro-symbolic RL with logical action-aware features)
therefore drives *which* polymer to evaluate next, while the evaluation and
scoring are identical across benchmark systems.

References
----------
1. IBM logical-agent-driven polymer discovery:
   https://github.com/IBM/logical-agent-driven-polymer-discovery
   Reinforcement Learning with Logical Action-Aware Features for Polymer
   Discovery. RL4RealLife @ ICML 2021.

2. Raffin, A., et al. (2021). Stable-Baselines3: Reliable Reinforcement
   Learning Implementations. *JMLR* 22(268):1–8.
   https://jmlr.org/papers/v22/20-1364.html

3. Mnih, V., et al. (2015). Human-level control through deep reinforcement
   learning. *Nature* 518, 529–533.

4. Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms.
   arXiv:1707.06347.

Usage
-----
.. code-block:: bash

    # Quick mock run (no OpenMM required)
    python benchmarks/ibm_insulin_rl_benchmark.py \\
        --mode train --algorithm dqn --n-timesteps 500 \\
        --mock --output results/ibm_dqn_mock.json

    # Real training with pre-computed cache
    python benchmarks/ibm_insulin_rl_benchmark.py \\
        --mode train --algorithm dqn --n-timesteps 50000 \\
        --cache-path data/ibm_psmiles_cache.json \\
        --output results/ibm_dqn.json

    # Test a saved model
    python benchmarks/ibm_insulin_rl_benchmark.py \\
        --mode test --algorithm dqn \\
        --model-path models/ibm_dqn_insulin.zip \\
        --n-episodes 20 --cache-path data/ibm_psmiles_cache.json \\
        --output results/ibm_dqn_test.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src" / "python"))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Shared comparison row schema
# ---------------------------------------------------------------------------
_COMPARISON_COLUMNS = [
    "method",
    "n_evaluations",
    "best_discovery_score",
    "best_interaction_energy_kj_mol",
    "n_high_performers_found",
    "n_unique_psmiles_evaluated",
    "wall_time_s",
    "algorithm",
    "n_timesteps_trained",
    "avg_episode_reward",
    "avg_targets_per_episode",
    "avg_steps_to_first_target",
    "avg_episode_length",
    "seed_psmiles",
    "n_proposals",
    "target_energy_kj",
    "notes",
]


def make_comparison_row(**kwargs: Any) -> Dict[str, Any]:
    """Return a dict with all comparison columns, filling missing with None."""
    return {col: kwargs.get(col) for col in _COMPARISON_COLUMNS}


def append_comparison_tsv(tsv_path: str, row: Dict[str, Any]) -> None:
    """Append one comparison row to a TSV file, creating header if needed."""
    p = Path(tsv_path)
    write_header = not p.is_file()
    with open(p, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COMPARISON_COLUMNS, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Mock evaluator (no OpenMM needed)
# ---------------------------------------------------------------------------
def _make_mock_evaluator(
    target_energy_kj: float = -5.0,
    random_seed: int = 42,
) -> Callable[[List[Dict[str, Any]], int], Dict[str, Any]]:
    """Return a deterministic mock evaluate_candidates_fn for testing."""
    import hashlib

    from insulin_ai.simulation.property_extractor import PropertyExtractor

    extractor = PropertyExtractor()

    def fn(candidates: List[Dict[str, Any]], max_candidates: int) -> Dict[str, Any]:
        md_results = []
        for c in candidates[:max_candidates]:
            ps = c.get("chemical_structure") or c.get("psmiles") or ""
            digest = int(hashlib.md5(ps.encode()).hexdigest()[:8], 16)
            # Deterministic fake energy: 30% chance of being a "target"
            fraction = (digest % 100) / 100.0
            e_int = target_energy_kj * 3 * fraction - 2.0  # range: [-17, -2]
            rmsd = 0.05 + 0.2 * fraction
            md_results.append({
                "psmiles": ps,
                "interaction_energy_kj_mol": e_int,
                "insulin_rmsd_to_initial_nm": rmsd,
                "potential_energy_complex_kj_mol": -1000.0 + e_int,
                "potential_energy_insulin_kj_mol": -800.0,
                "potential_energy_polymer_kj_mol": -200.0 + e_int * 0.1,
                "insulin_polymer_contacts": 8 if e_int < target_energy_kj else 2,
                "method": "mock",
            })
        names = [c.get("material_name", f"C_{i}") for i, c in enumerate(candidates[:max_candidates])]
        feedback = extractor.extract_feedback(md_results, names)
        feedback["md_results_raw"] = md_results
        return feedback

    return fn


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    algorithm: str,
    env_kwargs: Dict[str, Any],
    n_timesteps: int,
    model_path: Optional[str],
    random_seed: int,
    n_envs: int = 1,
    verbose: int = 0,
) -> Any:
    """Train a DQN or PPO policy on LogicalInsulinPSMILESEnv.

    Args:
        algorithm: ``"dqn"`` or ``"ppo"``.
        env_kwargs: Keyword arguments forwarded to ``LogicalInsulinPSMILESEnv``.
        n_timesteps: Total training steps.
        model_path: Optional path to save the trained model.
        random_seed: Seed for SB3 and the environment.
        n_envs: Parallel environments (PPO only; DQN ignores this).
        verbose: SB3 verbosity level (0=silent, 1=info).

    Returns:
        Trained SB3 model.
    """
    try:
        from stable_baselines3 import DQN, PPO
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError as e:
        raise ImportError(
            "stable-baselines3 is required for RL training. "
            "Install with: pip install stable-baselines3"
        ) from e

    from benchmarks.ibm_insulin_env import LogicalInsulinPSMILESEnv

    def make_env():
        return LogicalInsulinPSMILESEnv(**env_kwargs)

    if algorithm == "dqn":
        env = make_env()
        model = DQN(
            "MlpPolicy",
            env,
            gamma=0.8,
            learning_rate=3e-4,
            buffer_size=20_000,
            batch_size=32,
            seed=random_seed,
            verbose=verbose,
        )
    elif algorithm == "ppo":
        vec_env = make_vec_env(make_env, n_envs=n_envs, seed=random_seed)
        model = PPO(
            "MlpPolicy",
            vec_env,
            gamma=0.8,
            n_steps=512,
            learning_rate=3e-4,
            seed=random_seed,
            verbose=verbose,
        )
    else:
        raise ValueError(f"algorithm must be 'dqn' or 'ppo', got '{algorithm}'")

    logger.info(
        "Training %s for %d timesteps (n_envs=%d) …",
        algorithm.upper(),
        n_timesteps,
        n_envs,
    )
    model.learn(total_timesteps=n_timesteps)

    if model_path:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
        logger.info("Model saved to %s", model_path)

    return model


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

def test_model(
    model: Any,
    env_kwargs: Dict[str, Any],
    n_episodes: int,
    max_steps_per_episode: int = 100,
    vectorized_env: bool = False,
) -> Dict[str, Any]:
    """Run the trained model for n_episodes and collect metrics.

    Metrics mirror IBM upstream ``test_model``:
    - avg_reward, avg_targets, avg_episode_length, avg_steps_to_first_target
    Plus insulin-ai-specific:
    - best_discovery_score, best_interaction_energy_kj_mol,
      n_high_performers_found, all evaluated PSMILES
    """
    from benchmarks.ibm_insulin_env import LogicalInsulinPSMILESEnv
    from insulin_ai.simulation.scoring import discovery_score

    env = LogicalInsulinPSMILESEnv(**env_kwargs)

    log_rewards: List[float] = []
    log_targets: List[int] = []
    log_ep_lengths: List[int] = []
    log_first_target_steps: List[int] = []
    all_md_rows: List[Dict[str, Any]] = []
    all_target_psmiles: List[str] = []

    for episode in range(n_episodes):
        obs, _info = env.reset()
        ep_reward = 0.0
        targets = 0
        first_target_step: Optional[int] = None

        for step in range(max_steps_per_episode):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += float(reward)

            if info.get("tier") == "target":
                targets += 1
                if first_target_step is None:
                    first_target_step = step + 1

            md_row = info.get("md_row") or {}
            if md_row:
                md_row["episode"] = episode
                md_row["step"] = step
                md_row["tier"] = info.get("tier")
                all_md_rows.append(md_row)
                if info.get("tier") == "target":
                    all_target_psmiles.append(info.get("psmiles", ""))

            if terminated or truncated:
                break

        log_rewards.append(ep_reward)
        log_targets.append(targets)
        log_ep_lengths.append(step + 1)
        if first_target_step is not None:
            log_first_target_steps.append(first_target_step)

        logger.info(
            "Episode %d/%d: reward=%.3f targets=%d length=%d",
            episode + 1,
            n_episodes,
            ep_reward,
            targets,
            step + 1,
        )

    # Compute discovery metrics from all evaluated rows
    energies = [
        r["interaction_energy_kj_mol"]
        for r in all_md_rows
        if r.get("interaction_energy_kj_mol") is not None
    ]
    high_performers = [r["psmiles"] for r in all_md_rows if r.get("tier") == "target"]

    import numpy as np

    # Build feedback dict for discovery_score (same as MCP)
    pa = {}
    for r in all_md_rows:
        ps = r.get("psmiles", "")
        if ps:
            pa[ps] = r
    feedback_for_score = {
        "high_performers": list(dict.fromkeys(high_performers)),
        "effective_mechanisms": ["favorable_interaction_energy"] if high_performers else [],
        "problematic_features": [],
        "property_analysis": pa,
    }
    d_score = discovery_score(feedback_for_score)

    return {
        "n_episodes": n_episodes,
        "avg_episode_reward": float(np.mean(log_rewards)) if log_rewards else 0.0,
        "avg_targets_per_episode": float(np.mean(log_targets)) if log_targets else 0.0,
        "avg_episode_length": float(np.mean(log_ep_lengths)) if log_ep_lengths else 0.0,
        "avg_steps_to_first_target": (
            float(np.mean(log_first_target_steps)) if log_first_target_steps else None
        ),
        "best_discovery_score": round(d_score, 4),
        "best_interaction_energy_kj_mol": round(min(energies), 4) if energies else None,
        "n_high_performers_found": len(set(high_performers)),
        "n_unique_psmiles_evaluated": len({r.get("psmiles") for r in all_md_rows}),
        "n_evaluations": len(all_md_rows),
        "all_target_psmiles": list(dict.fromkeys(all_target_psmiles)),
        "energy_stats": {
            "mean": round(float(np.mean(energies)), 4) if energies else None,
            "min": round(float(min(energies)), 4) if energies else None,
            "max": round(float(max(energies)), 4) if energies else None,
            "n": len(energies),
        },
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_ibm_insulin_benchmark(
    mode: str = "train",
    algorithm: str = "dqn",
    seed_psmiles: str = "[*]OCC[*]",
    n_proposals: int = 20,
    max_steps: int = 100,
    n_targets: int = 5,
    target_energy_kj: float = -5.0,
    md_steps: int = 5000,
    n_timesteps: int = 50_000,
    n_episodes: int = 20,
    random_seed: int = 42,
    model_path: Optional[str] = None,
    cache_path: Optional[str] = None,
    mock: bool = False,
    comparison_tsv: Optional[str] = None,
    verbose: int = 0,
) -> Dict[str, Any]:
    """Full train-or-test pipeline for the IBM insulin RL benchmark.

    Args:
        mode: ``"train"``, ``"test"``, or ``"train_and_test"``.
        algorithm: ``"dqn"`` or ``"ppo"``.
        seed_psmiles: Initial PSMILES.
        n_proposals: Candidates proposed per step.
        max_steps: Max episode length.
        n_targets: Early-stop targets per episode.
        target_energy_kj: Reward "target" threshold (kJ/mol).
        md_steps: OpenMM minimisation steps (per candidate).
        n_timesteps: RL training steps.
        n_episodes: Test episodes.
        random_seed: Global RNG seed.
        model_path: Save / load model here.
        cache_path: Pre-computed evaluation cache (JSON).
        mock: Use mock evaluator instead of OpenMM (for CI / testing).
        comparison_tsv: If set, append a comparison row to this TSV file.
        verbose: SB3 verbosity (0=silent).

    Returns:
        JSON-serialisable result dict.
    """
    eval_fn = None
    if mock:
        eval_fn = _make_mock_evaluator(target_energy_kj=target_energy_kj, random_seed=random_seed)

    env_kwargs: Dict[str, Any] = dict(
        seed_psmiles=seed_psmiles,
        n_proposals=n_proposals,
        max_steps=max_steps,
        n_targets=n_targets,
        target_energy_kj=target_energy_kj,
        md_steps=md_steps,
        random_seed=random_seed,
        evaluate_candidates_fn=eval_fn,
        cache_path=cache_path,
    )

    t_start = time.perf_counter()
    result: Dict[str, Any] = {
        "algorithm": algorithm,
        "mode": mode,
        "seed_psmiles": seed_psmiles,
        "n_proposals": n_proposals,
        "target_energy_kj": target_energy_kj,
        "n_timesteps": n_timesteps,
        "n_episodes": n_episodes,
        "mock": mock,
        "cache_path": cache_path,
    }

    model = None

    if mode in ("train", "train_and_test"):
        model = train_model(
            algorithm=algorithm,
            env_kwargs=env_kwargs,
            n_timesteps=n_timesteps,
            model_path=model_path,
            random_seed=random_seed,
            verbose=verbose,
        )
        result["train_completed"] = True

    if mode == "test" or (mode == "train_and_test" and model is not None):
        if model is None:
            if not model_path:
                raise ValueError("--model-path required for --mode test")
            try:
                from stable_baselines3 import DQN, PPO
            except ImportError as e:
                raise ImportError("stable-baselines3 required") from e
            loader = DQN if algorithm == "dqn" else PPO
            model = loader.load(model_path)
            logger.info("Model loaded from %s", model_path)

        test_results = test_model(
            model=model,
            env_kwargs=env_kwargs,
            n_episodes=n_episodes,
            max_steps_per_episode=max_steps,
        )
        result.update(test_results)

    wall_time = time.perf_counter() - t_start
    result["wall_time_s"] = round(wall_time, 1)

    # Append to comparison TSV
    if comparison_tsv:
        row = make_comparison_row(
            method=f"ibm_rl_{algorithm}",
            n_evaluations=result.get("n_evaluations"),
            best_discovery_score=result.get("best_discovery_score"),
            best_interaction_energy_kj_mol=result.get("best_interaction_energy_kj_mol"),
            n_high_performers_found=result.get("n_high_performers_found"),
            n_unique_psmiles_evaluated=result.get("n_unique_psmiles_evaluated"),
            wall_time_s=round(wall_time, 1),
            algorithm=algorithm,
            n_timesteps_trained=n_timesteps if mode in ("train", "train_and_test") else None,
            avg_episode_reward=result.get("avg_episode_reward"),
            avg_targets_per_episode=result.get("avg_targets_per_episode"),
            avg_steps_to_first_target=result.get("avg_steps_to_first_target"),
            avg_episode_length=result.get("avg_episode_length"),
            seed_psmiles=seed_psmiles,
            n_proposals=n_proposals,
            target_energy_kj=target_energy_kj,
            notes="mock" if mock else ("cache" if cache_path else "live_openmm"),
        )
        append_comparison_tsv(comparison_tsv, row)
        logger.info("Comparison row appended to %s", comparison_tsv)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "IBM Logical-Agent RL benchmark adapted for insulin-ai evaluation.\n"
            "Uses MDSimulator + PropertyExtractor + discovery_score (same as MCP).\n"
            "Optimization loop: DQN / PPO with logical action-aware features (IBM)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["train", "test", "train_and_test"],
        default="train_and_test",
        help="Run mode (default: train_and_test).",
    )
    p.add_argument(
        "--algorithm",
        choices=["dqn", "ppo"],
        default="dqn",
        help="RL algorithm (default: dqn).",
    )
    p.add_argument(
        "--seed",
        type=str,
        default="[*]OCC[*]",
        dest="seed_psmiles",
        help="Seed PSMILES (default: PEG [*]OCC[*]).",
    )
    p.add_argument(
        "--n-proposals",
        type=int,
        default=20,
        help="Candidate pool size per step (action space, default: 20).",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum steps per episode (default: 100).",
    )
    p.add_argument(
        "--n-targets",
        type=int,
        default=5,
        help="Targets per episode before early stop (default: 5).",
    )
    p.add_argument(
        "--target-energy",
        type=float,
        default=-5.0,
        dest="target_energy_kj",
        help="Interaction energy threshold for 'target' reward kJ/mol (default: -5.0).",
    )
    p.add_argument(
        "--md-steps",
        type=int,
        default=5000,
        help="OpenMM minimisation steps per candidate (default: 5000).",
    )
    p.add_argument(
        "--n-timesteps",
        type=int,
        default=50_000,
        help="RL training timesteps (default: 50000).",
    )
    p.add_argument(
        "--n-episodes",
        type=int,
        default=20,
        help="Test episodes (default: 20).",
    )
    p.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Global RNG seed (default: 42).",
    )
    p.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to save (train) or load (test) the model.",
    )
    p.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help="Pre-computed PSMILES cache JSON (from precompute_psmiles_cache.py).",
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="Use mock evaluator (no OpenMM; for CI / quick tests).",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="JSON output path for full results.",
    )
    p.add_argument(
        "--comparison-tsv",
        type=str,
        default=None,
        help="Append a comparison row to this TSV (shared with optuna / agentic benchmarks).",
    )
    p.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="SB3 verbosity (0=silent, 1=info). Default: 0.",
    )
    args = p.parse_args()

    out = run_ibm_insulin_benchmark(
        mode=args.mode,
        algorithm=args.algorithm,
        seed_psmiles=args.seed_psmiles,
        n_proposals=args.n_proposals,
        max_steps=args.max_steps,
        n_targets=args.n_targets,
        target_energy_kj=args.target_energy_kj,
        md_steps=args.md_steps,
        n_timesteps=args.n_timesteps,
        n_episodes=args.n_episodes,
        random_seed=args.random_seed,
        model_path=args.model_path,
        cache_path=args.cache_path,
        mock=args.mock,
        comparison_tsv=args.comparison_tsv,
        verbose=args.verbose,
    )

    output_json = json.dumps(out, indent=2, default=str)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output_json)
        logger.info("Results written to %s", args.output)

    print(output_json)


if __name__ == "__main__":
    main()
