
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification
)

CONTRACT_TEXT = """
This Agreement is made between Acme Corp and Beta LLC on 1 Jan 2025.
The parties agree to keep all confidential information secret for a period of five years.
The liability of either party shall not exceed $100,000.
Either party may terminate this Agreement with 30 days written notice.
"""
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"


def get_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)


def get_classification_model(model_name, num_labels=5):
    return AutoModelForSequenceClassification.from_pretrained(model_name,
                                                              num_labels=num_labels)


def get_ner_model(model_name, num_labels=10):
    return AutoModelForTokenClassification.from_pretrained(model_name, 
                                                           num_labels=num_labels)

def classify_clauses(clauses, tokenizer, model):
    for clause in clauses:
        inputs = tokenizer(clause, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        pred_class = torch.argmax(outputs.logits, dim=1).item()
        print(f"Clause: {clause}")
        print(f"Predicted type: {pred_class}\n")


def extract_entities(clauses, tokenizer, model):
    for clause in clauses:
        tokens = tokenizer(clause, return_tensors="pt")
        outputs = model(**tokens)
        preds = torch.argmax(outputs.logits, dim=2)
        token_list = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
        entity_list = [model.config.id2label[p.item()] for p in preds[0]]
        print(f"Tokens: {token_list}")
        print(f"Entities: {entity_list}")
        print("\n")
        

if __name__ == "__main__":
    tokenizer = get_tokenizer(MODEL_NAME)
    seq_class_model = get_classification_model(MODEL_NAME)
    token_class_model = get_ner_model(MODEL_NAME)
    clauses = [c.strip() for c in CONTRACT_TEXT.split("\n") if c.strip()]

    print(token_class_model)

#    classify_clauses(clauses, tokenizer, seq_class_model)
#    extract_entities(clauses, tokenizer,  token_class_model)
