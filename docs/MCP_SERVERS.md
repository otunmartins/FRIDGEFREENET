# MCP Servers for Insulin AI

## Prerequisites

| Capability | Required |
|------------|----------|
| **`mine_literature`** | **Asta MCP** when `ASTA_API_KEY` set; else Semantic Scholar. |
| **`evaluate_psmiles`**, **`run_autonomous_discovery`** | **gmx** (GROMACS), **acpype**, **ambertools**, `data/4F1C.pdb`. Merged EM screening. |
| **Mutation** | psmiles / extras per `pyproject.toml`. |

## insulin-ai

| Command | `bash scripts/run_mcp_server.sh` |
|---------|----------------------------------|

**Session folder:** `runs/<id>/`. **Topology:** [GROMACS_TOPOLOGY.md](GROMACS_TOPOLOGY.md).

**GROMACS-VMD (integrated [mcp_gmx_vmd](../src/python/insulin_ai/mcp_gmx_vmd/)):** same MCP process, workspace `runs/gmx_vmd_workspace/`. Tools: **`gmx_vmd_create_workflow`**, **`gmx_vmd_list_workflows`**, **`gmx_vmd_gromacs_execute`** (run `gmx` in workflow dir), **`gmx_vmd_workflow_status`**, **`gmx_vmd_help`**, **`gmx_vmd_find_structures`**. Upstream: [gmx-vmd-mcp](https://github.com/egtai/gmx-vmd-mcp).
