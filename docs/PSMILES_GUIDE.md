# PSMILES (polymer SMILES) — guide for agents and humans

This file is the **canonical in-repo reference** for what PSMILES are, how `[*]` works, and how they relate (or do **not** relate) to material names like “PEG” or “PLA-PEG-PLA”. OpenCode agents using **materials-discovery** should treat this document as the definition to follow when proposing or checking structures.

## What PSMILES is

- **PSMILES** = polymer SMILES: a line notation for a **repeat unit** of a polymer, with **exactly two** connection points marked **`[*]`** (the stars attach to neighbors in the infinite chain).
- It is **not** a brand name, trade name, or block-copolymer acronym by itself. **“PLA-PEG-PLA”** in text does not automatically map to one correct PSMILES—you must choose a **chemistry-level repeat unit** (often simplified).

## Rules that validation enforces (syntax)

- The string must contain **`[*]`** (typically two stars for a linear backbone repeat unit).
- Packages (`psmiles`, RDKit) check **well-formedness** of the graph, not whether the string truly represents what a paper calls “chitosan”.

## Names vs structures (critical)

- **No automatic name→structure mapping** exists in this project. The **LLM** proposes PSMILES from names using chemistry knowledge; errors are possible.
- **Best practice:** for each candidate, keep a **table**: `material_name | PSMILES | source (paper / guess)` and use **`validate_psmiles`** with optional **`crosscheck_web`** (see MCP docs) to pull **web snippets** that may mention repeat units or SMILES—then **you** judge if they align.

## Common examples (repeat units, illustrative)

| Name (informal) | Example PSMILES (repeat unit) | Notes |
|-----------------|--------------------------------|--------|
| PEG / PEO | `[*]OCC[*]` | Poly(ethylene oxide); simplest repeat. |
| Polyethylene | `[*]CC[*]` | |
| Polylactide (PLA) | often simplified, e.g. lactide-derived repeat; structures vary | Use literature for the repeat you intend. |
| Polystyrene | `[*]CC([*])c1ccccc1` (variants exist) | |

Copolymers and block sequences usually need a **single repeat** that encodes your model’s intent, or **separate** homopolymer screens—not one ambiguous acronym.

## What simulation uses

- **`evaluate_psmiles`** builds an **oligomer** from your PSMILES, places it near insulin, and runs **OpenMM** minimization + interaction energy. The physics sees **only the PSMILES graph**, not the English name.

## Persistence in OpenCode

- This file lives in **`docs/PSMILES_GUIDE.md`**. It is **not** auto-injected into every model context; the **materials-discovery** agent is instructed to **read this file** when unsure. For a long session, the agent may re-read it or you can paste a short excerpt into the chat.

## Further reading

- Ramprasad-group **psmiles** tooling (canonicalization, etc.).
- Primary literature for each **specific** polymer’s repeat unit when accuracy matters.
