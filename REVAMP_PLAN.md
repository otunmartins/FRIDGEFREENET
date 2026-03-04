# Insulin AI Revamp Plan: Code-as-Platform for Polymeric Material Discovery

**Project**: AI-Driven Design of Fridge-Free Insulin Delivery Patches  
**Revision**: Complete revamp using code-as-platform paradigm  
**Reference**: [OpenCode.ai](https://opencode.ai) – code-first, composable development  
**Focus**: Impact feedback loop for polymeric insulin patch material discovery  
**Constraint**: **CPU-only execution** – no GPU dependencies

---

## 1. Executive Summary

This revamp transforms the Insulin AI project into a **code-as-platform** computational chemistry system that closes the active learning feedback loop for discovering polymeric materials for insulin patch delivery. The platform uses **PSMILES**, molecular simulation packages (OpenMM with PME, ASE), and runs entirely on **CPU** to maximize accessibility and reproducibility.

### Key Changes

| Aspect | Current | Revamp |
|--------|---------|--------|
| **Architecture** | Flat scripts, web-first | Modular `src/` layout, code-first |
| **MD Simulation** | Planned (UMA-ASE), not implemented | **OpenMM + PME + ASE**, CPU-only, functional |
| **Feedback Loop** | Mock/stub only | **Real MD evaluation** feeding literature mining |
| **Force Field** | UMA (GPU-biased) | **OpenMM Amber/GAFF** + optional **UMA-ASE** (CPU-capable) |
| **Project Layout** | Root-level Python files | `/src/python/`, `/tests/`, `/docs/`, `/benchmarks/` |

---

## 2. Code-as-Platform Philosophy

Following [OpenCode.ai](https://opencode.ai) and open computational chemistry ecosystems:

1. **Code is the primary interface** – CLI and Python APIs drive workflows; web UI is optional.
2. **Composable modules** – Literature mining, PSMILES generation, MD simulation, and feedback are separate, testable components.
3. **Reproducibility** – Fixed random seeds, versioned dependencies, deterministic pipelines.
4. **CPU-first** – No GPU required; use OpenMM `CPU` platform, NumPy-based analytics, and CPU-compatible force fields.
5. **Open tooling** – OpenMM, ASE, OpenFF, RDKit – no vendor lock-in.

---

## 3. Target Feedback Loop Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ACTIVE LEARNING FEEDBACK LOOP                         │
└─────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
  │  1. LITERATURE   │────▶│  2. PSMILES     │────▶│  3. MD          │
  │     MINING       │     │     GENERATION   │     │     SIMULATION  │
  │  (Semantic       │     │  (LLM + rules)   │     │  (OpenMM + PME)  │
  │   Scholar)       │     │                  │     │   CPU-only      │
  └────────┬─────────┘     └─────────────────┘     └────────┬────────┘
           │                                                 │
           │                                                  │
           │  ◀────────────── 4. FEEDBACK ◀──────────────────┘
           │       (high_performers, mechanisms, limitations)
           ▼
  ┌─────────────────┐
  │  DYNAMIC QUERIES │
  │  & PROMPTS      │
  └─────────────────┘
```

**Integration points**:
- Literature mining → material candidates (PSMILES-capable)
- PSMILES → 3D structure → MD input (OpenMM System)
- MD results → feedback dict → `mine_with_feedback()`
- Feedback → refined queries and prompts for next iteration

---

## 4. Project Structure (Post-Revamp)

```
insulin-ai/
├── src/
│   ├── python/                    # Python modules
│   │   ├── insulin_ai/
│   │   │   ├── __init__.py
│   │   │   ├── literature/         # Literature mining
│   │   │   │   ├── __init__.py
│   │   │   │   ├── miner.py
│   │   │   │   └── semantic_scholar_client.py
│   │   │   ├── psmiles/           # PSMILES generation
│   │   │   │   ├── __init__.py
│   │   │   │   └── generator.py
│   │   │   ├── simulation/        # MD pipeline (CPU-only)
│   │   │   │   ├── __init__.py
│   │   │   │   ├── openmm_runner.py   # OpenMM + PME
│   │   │   │   ├── psmiles_to_openmm.py # PSMILES → 3D → System
│   │   │   │   └── property_extractor.py # Thermal stability, etc.
│   │   │   ├── feedback/          # Active learning
│   │   │   │   ├── __init__.py
│   │   │   │   └── iterative_miner.py
│   │   │   └── chat/              # Chatbot, MCP (optional)
│   │   │       ├── __init__.py
│   │   │       ├── chatbot.py
│   │   │       └── ollama_client.py
│   │   └── app.py                 # Flask entry (thin)
│   └── native/                    # C++/Fortran/CUDA (future)
├── tests/
│   ├── test_literature.py
│   ├── test_psmiles.py
│   ├── test_simulation.py         # MD unit tests
│   └── test_feedback_loop.py      # Integration test
├── benchmarks/
│   └── md_benchmark.py            # CPU performance baseline
├── docs/
│   ├── REVAMP_PLAN.md             # This file
│   ├── method_notes.md            # Scientific methods
│   └── api_reference.md
├── mining_results/
├── iterative_results/
├── cycle_results/
├── requirements.txt
├── pyproject.toml                 # Modern Python packaging
└── README.md
```

---

## 5. CPU-Only Molecular Simulation Stack

### 5.1 Primary: OpenMM + PME

| Component | Role | CPU Support |
|-----------|------|-------------|
| **OpenMM** | MD engine, force fields | `Platform.getPlatformByName('CPU')` |
| **PME** | Long-range electrostatics | `NonbondedForce.PME` in OpenMM |
| **OpenFF / Amber** | Small molecule / polymer FF | GAFF, OpenFF via OpenMM |
| **RDKit** | PSMILES → 3D conformer | CPU-only |

### 5.2 Workflow Orchestration: ASE

- ASE `OpenMM` calculator or direct OpenMM Python API
- Use **OpenMM natively** for performance; ASE for structure conversion if needed
- **narupatools** or **ase-openmm** for ASE↔OpenMM bridge (optional)

### 5.3 Optional: UMA-ASE (CPU-Capable)

- UMA operates efficiently on CPU (MoLE ~50M active params)
- `uma-ase` or `fairchem-core` for ML force field
- Use as **alternative** to classical FF for validation

### 5.4 Explicitly Excluded

- CUDA, OpenCL platforms
- GPU-specific packages (e.g., CuPy)
- JAX/GPU backends

---

## 6. Implementation Phases

### Phase 1: Project Restructure ✅
- Create `src/python/insulin_ai/` package
- Move existing modules into subpackages
- Preserve backward compatibility with `app.py` entry point
- Add `pyproject.toml` for installable package

### Phase 2: MD Simulation Pipeline (CPU-Only)
- [x] `psmiles_to_openmm.py`: PSMILES → RDKit Mol → 3D → OpenMM System (GAFF when openff-toolkit available)
- [x] `openmm_runner.py`: Run NPT/NVT with PME, CPU platform
- [x] `property_extractor.py`: Energy-based stability metrics; RDKit proxy fallback
- [x] Unit tests with minimal polymer (e.g., PEG repeat)
- [x] `rdkit_proxy.py`: Fallback when GAFF unavailable

### Phase 3: Feedback Loop Integration
- [x] Implement `MDSimulator.evaluate_candidates(candidates)` interface
- [x] Wire `IterativeLiteratureMiner.run_active_learning_cycle(md_simulator=...)`
- [x] Map MD results → `high_performers`, `effective_mechanisms`, `problematic_features`
- [x] Integration test: MDSimulator + feedback dict format

### Phase 4: Dependencies & Documentation
- [x] `requirements.txt`: openmm, openmmforcefields, rdkit (openff-toolkit optional)
- [x] `docs/method_notes.md`: PME params, force field, units
- [x] RDKit proxy fallback when GAFF parameterization fails (e.g., openff-toolkit unavailable)

---

## 7. Key Technical Specifications

### 7.1 PME (Particle Mesh Ewald) Settings

```python
# OpenMM nonbonded method
nonbondedMethod = openmm.app.PME
nonbondedCutoff = 1.0 * unit.nanometers
ewaldErrorTolerance = 1e-4
```

### 7.2 Simulation Parameters (Insulin–Polymer Context)

- **Temperature**: 298 K (NPT) and 310 K (body temp)
- **Pressure**: 1 bar (NPT barostat)
- **Integrator**: Langevin (stochastic)
- **Timestep**: 2 fs
- **Minimal run**: 100 ps for screening; 1+ ns for validation

### 7.3 Units

- All quantities in **explicit units** (OpenMM `unit` module)
- Log constants and units in outputs per scientific reproducibility rules

---

## 8. Success Criteria

1. **Code runs 100% on CPU** – no CUDA/GPU imports or platform selection
2. **Closed feedback loop** – MD results meaningfully affect next literature iteration
3. **PSMILES → MD** – at least PEG and one copolymer flow end-to-end
4. **Tests pass** – pytest for literature, PSMILES, simulation, feedback
5. **Documentation** – method notes, API reference, CPU-only setup

---

## 9. Migration Path from Current Codebase

| Current File | New Location |
|--------------|--------------|
| `literature_mining_system.py` | `src/python/insulin_ai/literature/miner.py` |
| `semantic_scholar_client.py` | `src/python/insulin_ai/literature/semantic_scholar_client.py` |
| `psmiles_generator.py` | `src/python/insulin_ai/psmiles/generator.py` |
| `iterative_literature_mining.py` | `src/python/insulin_ai/feedback/iterative_miner.py` |
| `chatbot_system.py` | `src/python/insulin_ai/chat/chatbot.py` |
| `ollama_client.py` | `src/python/insulin_ai/chat/ollama_client.py` |
| `mcp_client.py`, `semantic_scholar_server.py` | `src/python/insulin_ai/chat/` |
| `app.py` | `src/python/app.py` (thin wrapper importing from package) |

---

## 10. References

- OpenCode.ai: https://opencode.ai  
- OpenMM User Guide: https://docs.openmm.org  
- ASE: https://wiki.fysik.dtu.dk/ase  
- UMA (Meta FAIR): https://ai.meta.com/research/publications/uma-a-family-of-universal-models-for-atoms  
- Proposal (proposal.tex): MD integration, UMA-ASE, ASE workflows  
- PSMILES: Sokolova et al., J. Chem. Inf. Model. (2021)

---

*Document version: 1.0 | Created: 2025-03-04*
