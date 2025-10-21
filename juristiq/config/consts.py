from pathlib import Path
from .common import DATA_DIR


# Folder where scripts output is stored.
_OUTPUT_DIR = Path(__file__).parent.parent / "output"
TEXT_ENCODING = "utf8"
# Prompt template used for verifying the contract's clause policy compliance.
CONTRACT_REVIEW_LLM_PROMPT_TEMPLATE_PATH = DATA_DIR / "clause_compliance_llm_prompt_template.yml"
# Prompt template used for generating multiple variations of an input text.
REPHRASE_TEXT_LLM_PROMPT_TEMPLATE_PATH = DATA_DIR / "rephrase_text_llm_prompt_template.yml"
# Prompt template used for generating multiple variations of combinations of 
# the rationale and the suggested redline for annotations with the "Missing" review label.
MISSING_CLAUSE_AUGMENTATION_LLM_PROMPT_TEMPLATE_PATH = DATA_DIR / "missing_clause_augmentation_llm_prompt_template.yml"
# Prompt template used for synthetic data generation for annotations that have the review label
# value set to "Compliant", "Risky"
CLAUSE_AUGMENTATION_LLM_PROMPT_TEMPLATE_PATH  = DATA_DIR / "clause_augmentation_llm_prompt_template.yml"
# Prompt template used for synthetic data generation for annotations that have the review label
# value set to "Lack of required data"
LACK_OF_REQUIRED_DATA_LLM_PROMPT_TEMPLATE = DATA_DIR / "lack_of_required_data_llm_prompt_template.yml"
# Prompt template used for synthetic data generation for annotations that have the review label
# value set to "Risky"
RISKY_CLAUSE_AUGMENTATION_LLM_PROMPT_TEMPLATE_PATH = DATA_DIR / "risky_llm_prompt_template.yml"
# Embedding models used when splitting contracts into chunks based on semantic similarity.
CONTRACTS_TEXT_EMBEDDINGS_MODEL = "nlpaueb/legal-bert-base-uncased"
# Sample of legal policies used for verifying contract clause compliance.
POLICIES_PATH = DATA_DIR /"policies.json"
# Path tp the folder with the augmented CUAD annotations.
TRANSFORMED_CUAD_ANNOTS_DIR_PATH = _OUTPUT_DIR / "juristiq-cuad-transformed"
# Path to the folder with the reviewed CUAD annotations regarding compliance policies.
REVIEWED_CUAD_ANNOTS_DIR_PATH = _OUTPUT_DIR / "juristiq-cuad-reviewed"
# Path to the folder where the reviewed CUAD annotations will be stored.
OUTPUT_DIR = _OUTPUT_DIR / "juristiq-cuad-reviewed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# Path to a file where tokens ratios of different tokenizer combinations are stored.
# Primarily used when calculating prompt tokens.
TOKENS_RATIO_FILE_PATH = DATA_DIR / "tokens_ratio.json"
