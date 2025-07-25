# Configuration for HuggingFace Spaces deployment

# App Configuration
APP_TITLE = "Insulin AI - Fridge-Free Delivery Patches"
APP_DESCRIPTION = "AI assistant for discovering materials for insulin patches that don't require refrigeration"
APP_VERSION = "1.0.0"

# Model Configuration
DEFAULT_MODEL = "microsoft/DialoGPT-medium"
FALLBACK_MODEL = "gpt2"

# UI Configuration
THEME = "soft"
MAX_TOKENS = 512
TEMPERATURE = 0.8  # Higher temperature for diverse candidate generation

# Example prompts for different modes
EXAMPLE_PROMPTS = {
    "general": [
        "What are the main challenges in insulin delivery?",
        "How do transdermal patches work?",
        "What makes insulin temperature sensitive?"
    ],
    "literature": [
        "Find papers on microneedle insulin delivery",
        "Search for hydrogel stability studies", 
        "Recent advances in polymer drug carriers"
    ],
    "psmiles": [
        "Generate a polymer for controlled release",
        "Create a pH-responsive hydrogel structure",
        "Design an adhesive patch polymer"
    ],
    "research": [
        "Franz diffusion cell methodology",
        "Insulin aggregation detection methods",
        "Biocompatibility testing protocols"
    ]
}

# Research domains covered
RESEARCH_DOMAINS = [
    "Insulin formulation and stability",
    "Transdermal drug delivery",
    "Polymer chemistry and materials science",
    "Pharmaceutical technology",
    "Biomedical engineering"
] 