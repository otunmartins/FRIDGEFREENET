# LLM-Guided SMILES Repair System

## Overview

This system implements an intelligent approach to fixing invalid SMILES strings by leveraging specific RDKit error messages and Large Language Model (LLM) reasoning. Instead of using generic repair methods, the system captures the exact error from RDKit and uses an LLM to understand and fix the specific problem.

## The Problem

When RDKit encounters an invalid SMILES string, it provides very specific error messages that tell us exactly what's wrong:

```
Example:
SMILES: "CC=OCCCNC"
Error: "Explicit valence for atom # 2 O, 3, is greater than permitted"
```

Previously, the system used generic repair methods that didn't take advantage of this specific diagnostic information.

## The Solution

### New Functions

#### 1. `fix_smiles_with_llm_guided_repair(smiles, error_message, model="granite3.3")`

This function takes a specific RDKit error message and uses an LLM to suggest targeted fixes.

**Features:**
- Provides context-specific prompts to the LLM
- Includes common error patterns and typical fixes
- Validates the LLM's suggestion before returning
- Returns detailed results with success/failure information

**Parameters:**
- `smiles`: The invalid SMILES string
- `error_message`: Specific RDKit error message
- `model`: Ollama model to use (default: "granite3.3")

**Returns:**
```python
{
    'success': bool,
    'fixed_smiles': str,  # if successful
    'explanation': str,   # if successful
    'original_smiles': str,
    'error': str          # if failed
}
```

#### 2. `capture_rdkit_error_and_repair(smiles, model="granite3.3")`

This function captures RDKit error messages and automatically attempts LLM-guided repair.

**Features:**
- Captures RDKit's stderr output to get specific error messages
- Handles both parsing errors (mol is None) and sanitization errors
- Automatically calls the LLM repair function if errors are detected
- Returns comprehensive results

**Parameters:**
- `smiles`: SMILES string to validate and potentially repair
- `model`: Ollama model to use for repairs

**Returns:**
```python
{
    'success': bool,
    'valid_smiles': str,     # if successful
    'message': str,          # description of what happened
    'original_smiles': str,  # if repair was needed
    'error_message': str,    # original RDKit error if repair was needed
    'error': str             # if failed
}
```

## Integration with Existing Workflow

The new system is integrated into the main SMILES validation workflow in `clean_psmiles_generation_with_langchain()`:

1. **Primary Method**: LLM-guided repair using specific error messages
2. **Fallback Methods**: Original repair methods (syntax fix, Datamol, SELFIES) if LLM repair fails

```python
# NEW APPROACH: Use LLM-guided repair with specific error messages first
repair_result = capture_rdkit_error_and_repair(smiles)

if repair_result['success']:
    # Use the repaired SMILES
    smiles = repair_result['valid_smiles']
else:
    # Fall back to original repair methods
    # ... existing code ...
```

## LLM Prompt Strategy

The system uses carefully crafted prompts that:

1. **Provide Context**: Include both the invalid SMILES and the specific error message
2. **Give Examples**: Show common error types and typical fixes
3. **Demand Precision**: Request only the corrected SMILES, no explanation
4. **Ensure Validation**: The response is validated with RDKit before acceptance

### Example Prompt

```
You are a chemistry expert specializing in SMILES notation repair. A SMILES string has failed validation with a specific error.

INVALID SMILES: CC=OCCCNC
ERROR MESSAGE: Explicit valence for atom # 2 O, 3, is greater than permitted

Please analyze this specific error and provide a corrected SMILES string. Focus on the exact issue mentioned in the error.

Common fixes for specific errors:
- "Explicit valence ... is greater than permitted": Fix atom valence by adjusting bonds or charges
- "Kekulization failure": Fix aromatic ring structure 
- "Ring closure" errors: Fix ring numbering or remove incomplete rings
- "Sanitization" errors: Fix chemical structure inconsistencies

Respond with ONLY the corrected SMILES string, nothing else. The corrected SMILES must be chemically valid.
```

## Error Types Handled

### 1. Valence Errors
- **Error**: "Explicit valence for atom # X Y, Z, is greater than permitted"
- **Common Fix**: Adjust bond orders or add/remove bonds to satisfy valence rules

### 2. Ring Closure Errors
- **Error**: "Ring closure" related messages
- **Common Fix**: Fix ring numbering, complete incomplete rings, or remove unmatched closures

### 3. Kekulization Failures
- **Error**: "Can't kekulize mol"
- **Common Fix**: Fix aromatic ring structures, ensure proper alternating bonds

### 4. Sanitization Errors
- **Error**: Various sanitization failure messages
- **Common Fix**: Fix chemical structure inconsistencies, proper charges, etc.

## Usage Examples

### Basic Usage

```python
from insulin_ai_app import capture_rdkit_error_and_repair

# Test with a problematic SMILES
result = capture_rdkit_error_and_repair("CC=OCCCNC")

if result['success']:
    print(f"Fixed SMILES: {result['valid_smiles']}")
    print(f"Original error was: {result['error_message']}")
else:
    print(f"Repair failed: {result['error']}")
```

### Direct LLM Repair

```python
from insulin_ai_app import fix_smiles_with_llm_guided_repair

smiles = "CC=OCCCNC"
error_msg = "Explicit valence for atom # 2 O, 3, is greater than permitted"

result = fix_smiles_with_llm_guided_repair(smiles, error_msg)

if result['success']:
    print(f"LLM suggested: {result['fixed_smiles']}")
    print(f"Explanation: {result['explanation']}")
```

## Testing

Run the test script to see the system in action:

```bash
python test_llm_repair.py
```

This will test various problematic SMILES strings and show how the LLM-guided repair works.

## Benefits

1. **Targeted Fixes**: Uses specific error information instead of generic approaches
2. **Higher Success Rate**: LLM can understand complex chemical problems
3. **Fallback Safety**: Still uses original methods if LLM repair fails
4. **Detailed Feedback**: Provides clear information about what was fixed and why
5. **Extensible**: Easy to add new error patterns and fixes

## Requirements

- Ollama with a capable model (granite3.3, llama3.2, etc.)
- RDKit for SMILES validation
- Python 3.7+ with standard libraries

## Future Enhancements

1. **Error Pattern Learning**: Build a database of successful repairs
2. **Model Selection**: Automatically choose the best model for different error types
3. **Confidence Scoring**: Rate the confidence of LLM suggestions
4. **Batch Processing**: Handle multiple SMILES repairs efficiently 