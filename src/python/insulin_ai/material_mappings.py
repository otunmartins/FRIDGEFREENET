#!/usr/bin/env python3
"""
PSMILES validation and cleaning via chemical informatics only.

No name→PSMILES mapping. The agent translates material names from its chemistry knowledge.
Validation and cleaning use only RDKit or Ramprasad's psmiles package — no static lookups.
"""


def validate_psmiles(psmiles: str) -> dict:
    """
    Validate PSMILES. Returns {valid: bool, canonical?: str, error?: str}.
    Uses psmiles.canonicalize when available, else RDKit.
    """
    if not psmiles or not isinstance(psmiles, str):
        return {"valid": False, "error": "Empty or invalid input"}

    psm = psmiles.strip()
    if "[*]" not in psm:
        return {"valid": False, "error": "PSMILES must contain [*] connection points"}

    # Try Ramprasad psmiles package (canonicalize)
    try:
        from psmiles import PolymerSmiles

        ps = PolymerSmiles(psm)
        # psmiles stable API: canonicalize is a property, not a method.
        c = ps.canonicalize
        if callable(c):
            c = c()
        canonical = str(c)
        return {"valid": True, "canonical": canonical}
    except ImportError:
        pass
    except Exception as e:
        return {"valid": False, "error": str(e)}

    # Fallback: RDKit validation (cap [*] to [H])
    try:
        from rdkit import Chem
        capped = psm.replace("[*]", "[H]")
        mol = Chem.MolFromSmiles(capped)
        if mol is None:
            return {"valid": False, "error": "Invalid SMILES structure"}
        return {"valid": True, "canonical": psm}
    except ImportError:
        return {"valid": False, "error": "rdkit or psmiles required for validation"}
    except Exception as e:
        return {"valid": False, "error": str(e)}


def clean_psmiles(psmiles: str) -> str | None:
    """
    Clean/repair PSMILES if possible. Returns canonical form or None.
    """
    r = validate_psmiles(psmiles)
    if r.get("valid"):
        return r.get("canonical", psmiles)
    return None
