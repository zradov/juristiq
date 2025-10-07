import os
from pathlib import Path

_DATA_DIR = Path(__file__).parent.parent / "data"
_OUTPUT_DIR = Path(__file__).parent.parent / "output"
# Prompt template used for verifying the contract's clause policy compliance.
CONTRACT_REVIEW_LLM_PROMPT_TEMPLATE_PATH = _DATA_DIR / "clause_compliance_llm_prompt_template.yml"
# Prompt template used for generating multiple variations of an input text.
REPHRASE_TEXT_LLM_PROMPT_TEMPLATE_PATH = _DATA_DIR / "rephrase_text_llm_prompt_template.yml"
# Prompt template used for generating multiple variations of combinations of 
# the rationale and the suggested redline for annotations with the "Missing" review label.
MISSING_CLAUSE_AUGMENTATION_LLM_PROMPT_TEMPLATE_PATH = _DATA_DIR / "missing_clause_augmentation_llm_prompt_template.yml"
# Prompt template used for synthetic data generation for annotations that have the review label
# value set to "Compliant", "Risky" or "Lack of required data".
CLAUSE_AUGMENTATION_LLM_PROMPT_TEMPLATE_PATH  = _DATA_DIR / "clause_augmentation_llm_prompt_template.yml"
# Embedding models used when splitting contracts into chunks based on semantic similarity.
CONTRACTS_TEXT_EMBEDDINGS_MODEL = "nlpaueb/legal-bert-base-uncased"
# Sample of legal policies used for verifying contract clause compliance.
POLICIES_PATH = _DATA_DIR /"policies.json"
# Path tp the folder with the augmented CUAD annotations.
TRANSFORMED_CUAD_ANNOTS_DIR_PATH = _OUTPUT_DIR / "juristiq-cuad-transformed"
# Path to the folder with the reviewed CUAD annotations regarding compliance policies.
REVIEWED_CUAD_ANNOTS_DIR_PATH = _OUTPUT_DIR / "juristiq-cuad-reviewed"
# Path to the folder where the reviewed CUAD annotations will be stored.
OUTPUT_DIR = _OUTPUT_DIR / "juristiq-cuad-reviewed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
